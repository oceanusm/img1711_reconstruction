/* Smooth clock correction. Never seek a decoder to correct ordinary drift. */
((root) => {
  'use strict';
  class PlaybackSync {
    constructor() { this.reset(); }
    reset() { this.filtered = null; this.lastTime = null; this.largeSince = null; }
    update(drift, now, rate) {
      const elapsed = this.lastTime === null ? 200 : Math.max(0, now - this.lastTime);
      this.lastTime = now;
      const alpha = 1 - Math.exp(-elapsed / 450);
      this.filtered = this.filtered === null ? drift : this.filtered + alpha * (drift - this.filtered);
      const correction = Math.abs(this.filtered) < .012 ? 0 : Math.max(-.05, Math.min(.05, this.filtered * .65));
      if (Math.abs(drift) > .35 && Math.abs(this.filtered) > .30) {
        if (this.largeSince === null) this.largeSince = now;
      } else this.largeSince = null;
      return {
        playbackRate: rate * (1 - correction),
        recover: this.largeSince !== null && now - this.largeSince >= 1500
      };
    }
  }
  if (typeof module === 'object' && module.exports) module.exports = PlaybackSync;
  else root.ComparisonPlaybackSync = PlaybackSync;
})(globalThis);
