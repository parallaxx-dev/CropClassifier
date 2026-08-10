// Plain HTML-table heatmap -- matches this site's existing plain-table
// aesthetic (see App.css) rather than pulling in a chart library for two
// static grids. Caller supplies a cellColor(value, row, col) function so the
// same component serves both a sequential (confusion matrix) and a
// diverging (band correlation) color scale.
export default function Heatmap({ labels, matrix, cellColor, cellText, rowLabel, colLabel }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', fontSize: 10, width: 'auto' }}>
        <thead>
          <tr>
            <th></th>
            {labels.map((l) => (
              <th
                key={l}
                style={{
                  writingMode: 'vertical-rl',
                  transform: 'rotate(180deg)',
                  padding: '2px 1px',
                  fontWeight: 400,
                  color: '#52514e',
                  verticalAlign: 'bottom',
                }}
              >
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th style={{ textAlign: 'left', paddingLeft: 2, paddingRight: 8, fontWeight: 400, color: '#52514e' }}>
                {labels[i]}
              </th>
              {row.map((v, j) => (
                <td
                  key={j}
                  title={`${rowLabel || ''} ${labels[i]} × ${colLabel || ''} ${labels[j]}: ${
                    cellText ? cellText(v, i, j) : v
                  }`}
                  style={{
                    background: cellColor(v, i, j),
                    width: 22,
                    height: 22,
                    minWidth: 22,
                    textAlign: 'center',
                    border: '1px solid #fcfcfb',
                    color: '#0b0b0b',
                    fontSize: 9,
                  }}
                >
                  {cellText ? cellText(v, i, j) : v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Sequential single-hue ramp (blue, light->dark) for non-negative magnitude
// data -- confusion-matrix cell counts/fractions.
const SEQUENTIAL_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

export function sequentialBlue(t) {
  // t in [0, 1]
  const clamped = Math.max(0, Math.min(1, t))
  const idx = Math.min(SEQUENTIAL_BLUE.length - 1, Math.floor(clamped * (SEQUENTIAL_BLUE.length - 1)))
  return SEQUENTIAL_BLUE[idx]
}

function lerp(a, b, t) {
  return Math.round(a + (b - a) * t)
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function rgbToHex([r, g, b]) {
  return `#${[r, g, b].map((x) => x.toString(16).padStart(2, '0')).join('')}`
}

// Diverging blue <-> red pair with a neutral gray midpoint, per the dataviz
// skill's color formula (two hues + neutral gray, equal steps per arm) --
// for polarity data (Pearson correlation, -1..+1).
const DIVERGING_RED = '#e34948'
const DIVERGING_GRAY = '#f0efec'
const DIVERGING_BLUE = '#256abf'

export function divergingBlueRed(t) {
  // t in [-1, 1]
  const clamped = Math.max(-1, Math.min(1, t))
  if (clamped < 0) {
    const rgb = hexToRgb(DIVERGING_RED).map((c, i) => lerp(hexToRgb(DIVERGING_GRAY)[i], c, -clamped))
    return rgbToHex(rgb)
  }
  const rgb = hexToRgb(DIVERGING_BLUE).map((c, i) => lerp(hexToRgb(DIVERGING_GRAY)[i], c, clamped))
  return rgbToHex(rgb)
}

export function ScaleLegend({ stops, labels }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, color: '#52514e', margin: '4px 0' }}>
      <span>{labels[0]}</span>
      <div
        style={{
          width: 160,
          height: 10,
          background: `linear-gradient(to right, ${stops.join(',')})`,
          border: '1px solid #c3c2b7',
        }}
      />
      <span>{labels[1]}</span>
    </div>
  )
}
