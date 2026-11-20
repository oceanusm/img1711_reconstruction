(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const config = window.COMPARISON;
  const videos = [...document.querySelectorAll('video')];
  const master = videos[0], figures = [...document.querySelectorAll('[data-video]')];
  const pts = config.pts, duration = config.duration;
  document.title = config.title || (config.scene || 'IMG_1696') + ' · Video comparison';
  $('title').textContent = document.title;
  $('subtitle').textContent = duration.toFixed(2) + ' seconds · ' + pts.length + ' source frames';
  $('timeline').max = duration;
  let mode = 'all', selected = $('render').value, rate = 1;
  let wanted = false, ready = false, operation = 0, buffering = false;
  let activeSeek = false, drag = false, dragResume = false, targetTime = 0;
  let seekSerial = Promise.resolve(), fatal = false;
  let lastMonitor = -Infinity, lastRecovery = -Infinity;
  const clocks = new Map(videos.map(v => [v, new window.ComparisonPlaybackSync()]));
  const activeVideos = () => mode === 'all' ? videos : [master, $(selected)];
  const clamp = t => Math.max(0, Math.min(pts[pts.length - 1] + .0005, Number(t) || 0));
  function stamp(t) {
    const ms = Math.max(0, Math.round(t * 1000));
    return String(Math.floor(ms / 60000)).padStart(2, '0') + ':' + String(Math.floor(ms / 1000) % 60).padStart(2, '0') + '.' + String(ms % 1000).padStart(3, '0');
  }
  function frameIndex(t) {
    let lo = 0, hi = pts.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (pts[mid] <= t + .0006) lo = mid + 1; else hi = mid; }
    return Math.max(0, lo - 1);
  }
  function message(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
  function ui() {
    $('play').textContent = !ready ? 'Loading…' : wanted ? 'Pause all' : 'Play in sync';
    $('play').setAttribute('aria-label', wanted ? 'Pause videos' : 'Play synchronized videos');
    if (!fatal) message(!ready ? 'Loading videos…' : activeSeek ? 'Seeking…' : buffering ? 'Buffering together…' : wanted ? 'Playing in sync' : 'Paused');
  }
  function showTime() {
    const t = drag ? Number($('timeline').value) : activeSeek ? targetTime : master.currentTime;
    if (!drag) $('timeline').value = t;
    $('current').textContent = stamp(t);
    $('timeline').setAttribute('aria-valuetext', stamp(t) + ' of ' + stamp(duration));
    $('frame').textContent = 'Frame ' + String(frameIndex(t) + 1).padStart(3, '0') + ' / ' + pts.length;
  }
  function pause() {
    wanted = false; operation++; buffering = false;
    videos.forEach(v => { v.pause(); v.playbackRate = rate; clocks.get(v).reset(); });
    ui(); showTime();
  }
  function waitSeek(video, time) {
    return new Promise((resolve, reject) => {
      if (Math.abs(video.currentTime - time) < .0005 && !video.seeking) return resolve();
      let timer;
      const cleanup = () => { clearTimeout(timer); video.removeEventListener('seeked', done); video.removeEventListener('error', fail); };
      const done = () => { cleanup(); resolve(); };
      const fail = () => { cleanup(); reject(new Error('A video could not seek.')); };
      video.addEventListener('seeked', done, {once:true}); video.addEventListener('error', fail, {once:true});
      timer = setTimeout(() => { cleanup(); reject(new Error('Seeking took too long. Try again.')); }, 12000);
      video.currentTime = time;
    });
  }
  function seek(time, resume = false) {
    const ticket = ++operation;
    targetTime = clamp(time); activeSeek = true; buffering = false; wanted = resume;
    videos.forEach(v => v.pause()); ui(); showTime();
    seekSerial = seekSerial.catch(() => {}).then(async () => {
      if (ticket !== operation) return;
      await Promise.all(activeVideos().map(v => waitSeek(v, targetTime)));
      if (ticket !== operation) return;
      activeSeek = false; ui(); showTime();
      if (resume) await startPlayback();
    }).catch(error => { if (ticket === operation) { activeSeek = false; pause(); message(error.message, true); } });
    return seekSerial;
  }
  async function startPlayback() {
    if (!ready || fatal || drag || activeSeek) return;
    const ticket = ++operation;
    wanted = true; buffering = false;
    activeVideos().forEach(v => { v.playbackRate = rate; clocks.get(v).reset(); }); ui();
    try {
      await Promise.all(activeVideos().map(v => v.play()));
      if (ticket !== operation && !wanted) videos.forEach(v => v.pause());
    } catch (error) {
      if (ticket !== operation) return;
      pause(); message(error.name === 'NotAllowedError' ? 'Press play to start the videos.' : 'Playback interrupted. Press play to retry.', true);
    }
  }
  function toggle() {
    if (!ready || fatal) return;
    if (wanted) pause();
    else seek(master.currentTime >= duration - .045 ? 0 : master.currentTime, true);
  }
  function stepFrame(delta) {
    if (!ready) return;
    const i = Math.max(0, Math.min(pts.length - 1, frameIndex(activeSeek ? targetTime : master.currentTime) + delta));
    seek(pts[i] + .0002, false);
  }
  function blend() {
    const value = Number($('blend').value);
    const top = figures.find(f => f.dataset.video === selected);
    top.style.opacity = mode === 'overlay' ? value / 100 : '';
    $('stage').style.setProperty('--wipe-position', value + '%');
    $('wipeLine').style.left = value + '%'; $('blendValue').value = value + '%';
  }
  function setView(nextMode = mode) {
    const resume = wanted, time = master.currentTime;
    pause(); mode = nextMode; selected = $('render').value;
    const layered = mode === 'overlay' || mode === 'wipe';
    $('stage').className = 'stage mode-' + mode;
    figures.forEach(f => {
      const id = f.dataset.video;
      f.hidden = mode !== 'all' && id !== 'original' && id !== selected;
      f.classList.toggle('base', layered && id === 'original');
      f.classList.toggle('top', layered && id === selected);
      f.style.opacity = '';
    });
    document.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    $('blendControls').hidden = !layered; $('overlayLabels').hidden = !layered; $('wipeLine').hidden = mode !== 'wipe';
    $('blendLabel').textContent = mode === 'wipe' ? 'Wipe position' : 'Render opacity';
    $('blend').setAttribute('aria-label', $('blendLabel').textContent);
    $('overlayName').textContent = $('render').selectedOptions[0].textContent;
    $('render').disabled = mode === 'all';
    sizeWipe(); blend();
    if (ready) seek(time, resume);
  }
  function sync(now) {
    // Media clocks and decoded frames update independently. Sample gently and
    // filter frame-sized jitter instead of repeatedly flushing the decoder.
    if (now - lastMonitor < 200) { requestAnimationFrame(sync); return; }
    lastMonitor = now;
    if (wanted && !activeSeek && !drag && !buffering) {
      const followers = activeVideos().filter(v => v !== master);
      let worst = 0;
      for (const v of followers) {
        const drift = v.currentTime - master.currentTime; worst = Math.max(worst, Math.abs(drift));
        const correction = clocks.get(v).update(drift, now, rate);
        if (Math.abs(v.playbackRate - correction.playbackRate) >= .002 * rate) v.playbackRate = correction.playbackRate;
        // A real, sustained clock failure is recovered once with both videos
        // paused and aligned, rather than seeking the moving right video.
        if (correction.recover && !v.seeking && now - lastRecovery > 5000) {
          lastRecovery = now; seek(master.currentTime, true); break;
        }
      }
      $('drift').textContent = 'Playback offset ≤ ' + Math.round(worst * 1000) + ' ms';
    } else $('drift').textContent = activeSeek ? 'Aligning frames' : 'Source timestamps';
    showTime(); requestAnimationFrame(sync);
  }
  function sizeWipe() {
    $('stage').style.setProperty('--stage-width', $('stage').getBoundingClientRect().width + 'px');
  }
  if (window.ResizeObserver) new ResizeObserver(sizeWipe).observe($('stage'));
  else window.addEventListener('resize', sizeWipe);
  $('play').addEventListener('click', toggle);
  $('restart').addEventListener('click', () => seek(0, wanted));
  $('prev').addEventListener('click', () => stepFrame(-1)); $('next').addEventListener('click', () => stepFrame(1));
  $('speed').addEventListener('change', () => { rate = Number($('speed').value); videos.forEach(v => { v.playbackRate = rate; clocks.get(v).reset(); }); });
  $('audio').addEventListener('click', () => { master.muted = !master.muted; $('audio').textContent = master.muted ? 'Audio off' : 'Original audio'; $('audio').setAttribute('aria-pressed', String(!master.muted)); });
  $('fullscreen').addEventListener('click', async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else if ($('app').requestFullscreen) await $('app').requestFullscreen(); else message('Use your browser’s fullscreen command.'); } catch { message('Use your browser’s fullscreen command.'); }
  });
  document.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => setView(b.dataset.mode)));
  $('render').addEventListener('change', () => setView()); $('blend').addEventListener('input', blend);
  $('timeline').addEventListener('pointerdown', () => { dragResume = wanted; pause(); drag = true; });
  $('timeline').addEventListener('input', () => {
    if (!drag) { dragResume = wanted; pause(); drag = true; }
    seek(Number($('timeline').value), false);
  });
  function endDrag() { if (!drag) return; const time = Number($('timeline').value); drag = false; const resume = dragResume; dragResume = false; seek(time, resume); }
  $('timeline').addEventListener('change', endDrag); window.addEventListener('pointerup', endDrag); window.addEventListener('pointercancel', endDrag); $('timeline').addEventListener('blur', endDrag);
  document.addEventListener('keydown', event => {
    if (/INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.code === 'Space') { event.preventDefault(); toggle(); }
    else if (event.key === ',' || event.key === '.') { event.preventDefault(); stepFrame(event.key === ',' ? -1 : 1); }
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault(); const direction = event.key === 'ArrowLeft' ? -1 : 1;
      if (event.shiftKey) stepFrame(direction); else seek((activeSeek ? targetTime : master.currentTime) + direction, wanted);
    }
  });
  for (const v of videos) {
    v.addEventListener('error', () => {
      fatal = true; pause();
      const box = v.parentElement.querySelector('.video-error'); box.hidden = false;
      box.textContent = 'Video could not load. Keep the media folder beside index.html and open the extracted page in Safari or Chrome.';
      message('A video is unavailable. Check the media folder.', true);
    });
    v.addEventListener('waiting', () => {
      if (!wanted || activeSeek || drag || !activeVideos().includes(v)) return;
      buffering = true; operation++; activeVideos().forEach(item => item.pause()); ui();
    });
    v.addEventListener('canplay', () => {
      if (wanted && buffering && activeVideos().every(item => item.readyState >= 3)) {
        buffering = false; seek(master.currentTime, true);
      }
    });
  }
  master.addEventListener('ended', () => { pause(); message('End of recording · press play to replay'); });
  Promise.all(videos.map(v => v.readyState >= 2 ? Promise.resolve() : new Promise(resolve => v.addEventListener('loadeddata', resolve, {once:true})))).then(() => {
    if (fatal) return;
    ready = true; ['play','restart','prev','next','timeline'].forEach(id => { $(id).disabled = false; });
    $('duration').textContent = stamp(duration); ui(); showTime();
  });
  setView();
  window.addEventListener('pagehide', pause);
  requestAnimationFrame(sync);
})();
