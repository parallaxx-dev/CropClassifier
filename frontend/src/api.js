// Thin fetch wrappers around the FastAPI backend. No axios -- plain fetch is
// enough for this many endpoints and keeps the dependency list small.

// VITE_API_BASE is a build-time env var (Vite inlines import.meta.env.* into
// the bundle at build time, not read at container runtime) -- set it when
// building the frontend image so it points at wherever the backend actually
// runs (an ECS service's ALB URL in production), instead of hardcoding the
// local dev backend port. Falls back to local dev when unset.
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:9000/api/v1'

async function get(path) {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`GET ${path} failed (${res.status}): ${body}`)
  }
  return res.json()
}

export function fetchModelInfo() {
  return get('/model-info')
}

export function fetchRegions() {
  return get('/parcels/regions')
}

export function fetchParcels(region, limit = 40) {
  const params = new URLSearchParams()
  if (region) params.set('region', region)
  params.set('limit', String(limit))
  return get(`/parcels?${params.toString()}`)
}

export function fetchDemoParcels() {
  return get('/demo/parcels')
}

export async function repredictDemoParcels() {
  const res = await fetch(`${API_BASE}/demo/repredict`, { method: 'POST' })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`repredict failed (${res.status}): ${body}`)
  }
  return res.json()
}

export async function uploadAoiFile(file) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_BASE}/upload-aoi`, { method: 'POST', body: formData })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`upload failed (${res.status}): ${body}`)
  }
  return res.json()
}

export async function predict({ aoi, startDate, endDate }) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      aoi,
      start_date: startDate,
      end_date: endDate,
    }),
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`predict failed (${res.status}): ${body}`)
  }
  return res.json()
}
