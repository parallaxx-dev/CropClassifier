// Parses free-typed/pasted coordinate input into a GeoJSON Polygon geometry.
// Tries, in order: GeoJSON (bare geometry or a Feature wrapping one), WKT
// POLYGON((...)), then one "lon,lat" (or "lon lat") pair per line. Returns
// null if none of these parse -- callers show a single generic error rather
// than trying to explain which of three grammars failed.
export function parsePolygonInput(text) {
  const trimmed = text.trim()
  if (!trimmed) return null

  try {
    const json = JSON.parse(trimmed)
    const geom = json.type === 'Feature' ? json.geometry : json
    if (geom && geom.type === 'Polygon' && Array.isArray(geom.coordinates)) {
      return geom
    }
  } catch {
    // not JSON -- fall through to the other grammars
  }

  const wktMatch = trimmed.match(/POLYGON\s*\(\(([^)]+)\)\)/i)
  if (wktMatch) {
    const pairs = wktMatch[1].split(',').map((pair) => pair.trim().split(/\s+/).map(Number))
    return ringToPolygon(pairs)
  }

  const lines = trimmed.split(/\r?\n/).map((l) => l.trim()).filter(Boolean)
  const pairs = lines.map((line) => line.split(/[,\s]+/).map(Number))
  if (pairs.length >= 3 && pairs.every((p) => p.length === 2 && p.every((n) => !Number.isNaN(n)))) {
    return ringToPolygon(pairs)
  }

  return null
}

// Mirrors backend/app/services/validation.py's approx_area_hectares exactly
// (same lat-corrected planar shoelace approximation) so a drawn/pasted AOI
// shows the same area client-side, before submission, that the backend
// would compute for it -- "good enough for a sanity bound," not a precise
// geodesic calculation, same caveat as the Python original.
export function approxAreaHectares(geometry) {
  const ring = geometry.coordinates[0]
  const lat = ring.reduce((sum, [, lat]) => sum + lat, 0) / ring.length
  const metersPerDegreeLon = 111_320 * Math.cos((lat * Math.PI) / 180)
  const metersPerDegreeLat = 110_540

  let areaDeg2 = 0
  for (let i = 0; i < ring.length - 1; i++) {
    const [x1, y1] = ring[i]
    const [x2, y2] = ring[i + 1]
    areaDeg2 += x1 * y2 - x2 * y1
  }
  areaDeg2 = Math.abs(areaDeg2) / 2

  return (areaDeg2 * metersPerDegreeLon * metersPerDegreeLat) / 10_000
}

function ringToPolygon(pairs) {
  if (pairs.length < 3 || pairs.some(([lon, lat]) => Number.isNaN(lon) || Number.isNaN(lat))) return null
  const ring = [...pairs]
  const [firstLon, firstLat] = ring[0]
  const [lastLon, lastLat] = ring[ring.length - 1]
  if (firstLon !== lastLon || firstLat !== lastLat) ring.push([firstLon, firstLat])
  return { type: 'Polygon', coordinates: [ring] }
}
