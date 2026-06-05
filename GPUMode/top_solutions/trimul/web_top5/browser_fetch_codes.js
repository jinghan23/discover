// Run this in the browser DevTools console on https://www.gpumode.com/leaderboard/496?tab=rankings
// while logged in. It downloads the code payload for the submissions worth studying locally:
// A100 top5 for A800/sm80 work, H100 top5 for cross-checking ideas, and B200 top2 for optional reference.
const payload = {
  leaderboard_id: 496,
  submission_ids: [
    782275, 781115, 380716, 781360, 483089,
    782080, 781381, 450489, 408928, 781100,
    782380, 480316,
  ]
};
const response = await fetch('/api/codes', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  credentials: 'include',
  body: JSON.stringify(payload),
});
const data = await response.json();
console.log(data);
const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'gpumode496_a100_h100_b200_selected_codes.json';
a.click();
URL.revokeObjectURL(url);
