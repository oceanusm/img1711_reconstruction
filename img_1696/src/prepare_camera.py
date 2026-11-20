"""Prepare one camera sample per decoded frame using the matching VGGT archive.

Usage: python prepare_camera.py /path/src/img_1696.mov [output_camera_track.npz]
Dependencies: FFmpeg/ffprobe on PATH, Python with NumPy and OpenCV.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def source_paths(video):
    match = re.fullmatch(r'img_(\d+)', video.stem, re.I)
    if not match:
        raise ValueError('Expected img_<ID>.mov')
    return video.parent / (match.group(1) + '_vggt')


def read_cloud(path):
    dtype = np.dtype([('xyz', '<f4', (3,)), ('rgb', 'u1', (3,)),
                      ('confidence', '<f4'), ('frame_index', '<i4')])
    with path.open('rb') as f:
        header = []
        while True:
            line = f.readline(5000)
            if line.strip() == b'end_header':
                break
            if not line:
                raise ValueError('Incomplete PLY')
            header.append(line.decode('ascii').strip())
        expected = ['property float x', 'property float y', 'property float z',
                    'property uchar red', 'property uchar green', 'property uchar blue',
                    'property float confidence', 'property int frame_index']
        if [x for x in header if x.startswith('property ')] != expected:
            raise ValueError('Unexpected VGGT PLY schema')
        if 'format binary_little_endian 1.0' not in header:
            raise ValueError('Expected little-endian PLY')
        return np.memmap(path, mode='r', dtype=dtype, offset=f.tell())


def affine_resize(K, scale):
    # Exact half-pixel-center convention used by the input preprocessing.
    A = np.diag([scale[0], scale[1], 1.0])
    A[:2, 2] = (scale - 1) / 2
    return A @ K


def interpolate_pose(p0, p1, t):
    axis = cv2.Rodrigues(p0[:3, :3].T @ p1[:3, :3])[0]
    p = np.eye(4)
    p[:3, :3] = p0[:3, :3] @ cv2.Rodrigues(axis*t)[0]
    p[:3, 3] = p0[:3, 3]*(1-t) + p1[:3, 3]*t
    return p


def depth_features(image, xyz, c2w, K):
    xy = cv2.goodFeaturesToTrack(image, 380, 0.010, 8, blockSize=5)
    if xy is None:
        return np.empty((0, 2), np.float32), np.empty((0, 3))
    xy = xy[:, 0].astype(np.float32)
    cam = np.einsum('ni,ij->nj', np.asarray(xyz, dtype=np.float64) - c2w[:3, 3], c2w[:3, :3])
    cam = cam[np.isfinite(cam).all(1) & (cam[:, 2] > 0.001)]
    uvw = np.einsum('ni,ji->nj', cam, K)
    projected = (uvw[:, :2]/uvw[:, 2:]).astype(np.float32)
    if len(projected) < 4:
        return np.empty((0, 2), np.float32), np.empty((0, 3))
    tree = cv2.flann_Index(projected, dict(algorithm=1, trees=2))
    ids, d2 = tree.knnSearch(xy, 4, params={'checks': 32})
    depths = cam[ids, 2]
    keep = (d2[:, 0] < 4**2) & (d2[:, -1] < 13**2)
    keep &= (depths.max(1) - depths.min(1)) < 0.10*np.median(depths, axis=1)
    xy = xy[keep]
    # Back-project features using the nearest reliable depth sample.
    ray = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(K).T
    xyz = np.einsum('ni,ji->nj', ray*depths[keep, :1], c2w[:3, :3]) + c2w[:3, 3]
    return xy, xyz


def track(im0, im1, xy):
    if not len(xy):
        return xy, np.zeros(0, dtype=bool)
    settings = dict(winSize=(19, 19), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 22, .02))
    nxt, ok, err = cv2.calcOpticalFlowPyrLK(im0, im1, xy, None, **settings)
    previous, back, _ = cv2.calcOpticalFlowPyrLK(im1, im0, nxt, None, **settings)
    good = (ok.ravel() != 0) & (back.ravel() != 0)
    good &= np.linalg.norm(previous - xy, axis=1) < .6
    good &= (err.ravel() < 20)
    h, w = im1.shape
    good &= (nxt[:, 0] > 5) & (nxt[:, 0] < w - 5) & (nxt[:, 1] > 5) & (nxt[:, 1] < h - 5)
    return nxt, good


def project(c2w, K, xyz):
    cam = np.einsum('ni,ij->nj', xyz - c2w[:3, 3], c2w[:3, :3])
    uvw = np.einsum('ni,ji->nj', cam, K)
    return uvw[:, :2]/uvw[:, 2:]


def refine_pose(initial, K, X, uv, rng):
    if len(X) < 40:
        return initial, False, None
    ids = rng.permutation(len(X))
    valid, fit = ids[:len(ids)//4], ids[len(ids)//4:]
    extr = np.linalg.inv(initial)
    rot = cv2.Rodrigues(extr[:3, :3])[0]
    translation = extr[:3, 3].copy()
    ok, rot, translation, inliers = cv2.solvePnPRansac(
        X[fit], uv[fit], K, None, rot, translation, useExtrinsicGuess=True,
        iterationsCount=75, reprojectionError=1.8, confidence=.99,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers) < .55 * len(fit):
        return initial, False, None
    fit_inliers = fit[inliers.ravel()]
    rot, translation = cv2.solvePnPRefineLM(X[fit_inliers], uv[fit_inliers],
                                             K, None, rot, translation)
    extr[:3, :3] = cv2.Rodrigues(rot)[0]
    extr[:3, 3] = translation.ravel()
    candidate = np.linalg.inv(extr)
    # Hold-out tracks test whether the measured displacement improves the interpolant.
    e0 = np.linalg.norm(project(initial, K, X[valid]) - uv[valid], axis=1)
    e1 = np.linalg.norm(project(candidate, K, X[valid]) - uv[valid], axis=1)
    good = np.median(e1) + .025 < np.median(e0)
    good &= np.linalg.norm(initial[:3, 3] - candidate[:3, 3]) < .025
    good &= np.linalg.norm(cv2.Rodrigues(initial[:3, :3].T @ candidate[:3, :3])[0]) < .035
    return (candidate if good else initial), bool(good), (e0, e1 if good else e0)


def timing(video, cache=None):
    from fractions import Fraction
    props = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=time_base,r_frame_rate,avg_frame_rate,width,height:stream_side_data=rotation',
            '-of','json',str(video)]))['streams'][0]
    if cache:
        lines = cache.read_text().splitlines()
    else:
        lines = subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                 '-show_entries','frame=pts,duration','-of','csv=p=0',str(video)],
                 text=True).splitlines()
    pts = np.array([int(s.split(',')[0]) for s in lines if s and s[0].isdigit()], dtype=np.int64)
    return pts, float(Fraction(props['time_base'])), props


def prepare(video, outfile, gray_file, frame_times):
    ref = source_paths(video)
    idx = json.loads((ref/'index.json').read_text())
    data = np.load(ref/'cameras.npz', allow_pickle=False)
    anchors = data['frame_indices']
    poses = data['camera_to_world'].astype(float)
    original_K = data['intrinsics_canonical']
    indexed = sorted(idx['frames'], key=lambda f: f['camera_array_index'])
    assert np.array_equal([f['index'] for f in indexed], anchors)
    pts, time_base, props = timing(video, frame_times)
    num = len(pts)
    w, h = idx['preprocessing']['canonical_size_wh']
    sw = int(round(w * 640/max(w, h)/2)*2)
    sh = int(round(h * 640/max(w, h)/2)*2)
    scale = np.array([sw/w, sh/h])
    if max(abs(pts[anchors]*time_base - data['pts_seconds'])) > .001:
        raise ValueError('Source PTS do not correspond to the camera archive')
    if (anchors[0], anchors[-1]) != (0, num-1):
        raise ValueError('Reference does not cover both endpoints')
    if gray_file is None:
        temp = tempfile.NamedTemporaryFile()
        gray_file = Path(temp.name)
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),
            '-map','0:v:0','-vf',f'scale={sw}:{sh}:flags=area,format=gray',
            '-fps_mode','passthrough','-f','rawvideo','-y',str(gray_file)], check=True)
    if gray_file.stat().st_size != sw * sh * num:
        raise ValueError('Decoded image count differs from the probed timestamp count')
    gray = np.memmap(gray_file, mode='r', dtype='u1', shape=(num,sh,sw))
    cloud = read_cloud(ref/'point_cloud.ply')
    frame_map = cloud['frame_index'].copy()
    if np.any(np.diff(frame_map) < 0):
        order = np.argsort(frame_map, kind='stable')
        cloud = cloud[order]
        frame_map = frame_map[order]
    curves = np.empty((num,4,4))
    intrinsics = np.empty((num,3,3))
    mode = np.zeros(num, np.uint8)
    rng = np.random.default_rng(16960)
    cv2.setRNGSeed(16960)
    cv2.setNumThreads(2)
    # Blender renders square pixels. Symmetrize the slightly unequal estimated focal lengths.
    Ksq = original_K.copy()
    Ksq[:, 0, 0] = Ksq[:, 1, 1] = np.sqrt(original_K[:,0,0]*original_K[:,1,1])
    curves[anchors] = poses
    intrinsics[anchors] = Ksq
    smallK = [affine_resize(k, scale) for k in Ksq]
    initial_errors, final_errors, anchor_errors = [], [], []
    track_counts = []
    for i, a in enumerate(anchors):
        lo, hi = np.searchsorted(frame_map, [a, a+1])
        # Use the actual source camera intrinsics when assigning depths to image features.
        xy, xyz = depth_features(gray[a], cloud['xyz'][lo:hi], poses[i], affine_resize(original_K[i], scale))
        if i:
            b, a0 = a, anchors[i-1]
            for f in range(a0+1, b):
                t = (pts[f] - pts[a0])/(pts[b] - pts[a0])
                k = (1-t)*smallK[i-1] + t*smallK[i]
                intrinsics[f] = (1-t)*Ksq[i-1] + t*Ksq[i]
                uv_a, ok_a = track(gray[a0], gray[f], prev_xy)
                uv_b, ok_b = track(gray[b], gray[f], xy)
                obs = np.vstack([uv_a[ok_a], uv_b[ok_b]])
                landmarks = np.vstack([prev_xyz[ok_a], xyz[ok_b]])
                prior = interpolate_pose(poses[i-1], poses[i], t)
                curves[f], improved, errors = refine_pose(prior, k, landmarks, obs, rng)
                mode[f] = 2 if improved else 1
                if errors is not None:
                    e0, e1 = errors
                    initial_errors.extend(e0/scale.mean())
                    final_errors.extend(e1/scale.mean())
                track_counts.append(len(obs))
            across, ok = track(gray[a0], gray[b], prev_xy)
            err = np.linalg.norm(project(poses[i], smallK[i], prev_xyz[ok])-across[ok],axis=1)
            anchor_errors.extend(err/scale.mean())
        prev_xy, prev_xyz = xy, xyz
    def summary(v):
        return dict(count=len(v), median=float(np.median(v)), p90=float(np.percentile(v,90)))
    stats = {'tracking_image_dimensions':[sw,sh], 'error_units':'oriented source pixels',
             'holdout_prior_reprojection_px':summary(initial_errors),
             'holdout_final_reprojection_px':summary(final_errors),
             'adjacent_anchor_reprojection_px':summary(anchor_errors),
             'matched_tracks_per_intermediate_frame':summary(track_counts),
             'reference_anchors':len(anchors), 'optical_flow_refined':int((mode==2).sum()),
             'interpolated':int((mode==1).sum()),
             'focal_symmetrization_max_fraction':float(np.max(abs(Ksq[:,:2,:2].diagonal(axis1=1,axis2=2)/
                    original_K[:,:2,:2].diagonal(axis1=1,axis2=2)-1)))}
    outfile.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(outfile, camera_to_world_cv=curves, intrinsics=intrinsics, pts=pts,
        time_base=time_base, mode=mode, size_wh=[w,h], source_name=video.name,
        reference_sha256=hashlib.sha256((ref/'cameras.npz').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        stream_info=json.dumps(props), stats=json.dumps(stats))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path, nargs='?')
    args = parser.parse_args()
    video=args.video.expanduser().resolve()
    target=args.output or video.parent.parent/'output'/video.stem/'src'/'camera_track.npz'
    prepare(video,target,args.gray_cache,args.pts_cache)
