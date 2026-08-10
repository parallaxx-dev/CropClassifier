import { useEffect, useRef, useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { MapContainer, GeoJSON, useMap } from 'react-leaflet'
import L from 'leaflet'
import { fetchRegions, fetchParcels, predict } from '../api'
import BaseLayers from '../components/BaseLayers'

// AgriFieldNet (India) was fetched against 2024 imagery (see
// eurocrops_pipeline/agrifieldnet.py); every EuroCrops country uses its own
// declared year (see eurocrops_pipeline/config.py's COUNTRIES table). This is
// just a sane default for the date inputs -- the user can change it.
const DEFAULT_YEAR = {
  AT: 2021,
  BE_VLG: 2021,
  CZ: 2023,
  DE_BB: 2023,
  DE_LS: 2021,
  DE_NRW: 2021,
  DK: 2019,
  BZH: 2017,
  IN: 2024,
  // CUSTOM: hand-labeled parcels (see app/services/custom_parcels.py) whose
  // Sentinel-2 time series was actually fetched for full calendar year 2025
  // (training/eurocrops_pipeline/run_custom.py's CUSTOM_YEAR) -- must match
  // that, not just "a recent year", or the default date range silently
  // queries a year the model was never trained/fetched against.
  CUSTOM: 2025,
}

function FitBounds({ geojson }) {
  const map = useMap()
  useEffect(() => {
    if (!geojson || geojson.features.length === 0) return
    const layer = L.geoJSON(geojson)
    const bounds = layer.getBounds()
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [20, 20] })
    }
  }, [geojson, map])
  return null
}

// Real sample parcels are spread across an entire country, not clustered --
// fitting to ALL of them zooms out so far that individual field boundaries
// (a few hundred meters across) are sub-pixel and unclickable. Picking a
// parcel from the list zooms tight to that one field instead, the way a real
// GIS "zoom to feature" tool would.
function ZoomToSelected({ feature }) {
  const map = useMap()
  useEffect(() => {
    if (!feature) return
    const bounds = L.geoJSON(feature).getBounds()
    if (bounds.isValid()) {
      map.flyToBounds(bounds, { padding: [80, 80], maxZoom: 17, duration: 0.5 })
    }
  }, [feature, map])
  return null
}

function parcelStyle(feature, selectedId) {
  const isSelected = feature.properties.id === selectedId
  return {
    color: isSelected ? '#000000' : feature.properties.color,
    weight: isSelected ? 4 : 2,
    fillColor: feature.properties.color,
    fillOpacity: isSelected ? 0.35 : 0.15,
  }
}

export default function ParcelBrowser() {
  const [region, setRegion] = useState('AT')
  const [selected, setSelected] = useState(null)
  const [startDate, setStartDate] = useState(`${DEFAULT_YEAR.AT}-01-01`)
  const [endDate, setEndDate] = useState(`${DEFAULT_YEAR.AT}-12-31`)
  const geoJsonRef = useRef(null)

  const regionsQuery = useQuery({ queryKey: ['regions'], queryFn: fetchRegions })
  const parcelsQuery = useQuery({
    queryKey: ['parcels', region],
    queryFn: () => fetchParcels(region, 40),
    enabled: !!region,
  })

  const predictMutation = useMutation({ mutationFn: predict })

  function handleRegionChange(e) {
    const r = e.target.value
    setRegion(r)
    setSelected(null)
    predictMutation.reset()
    const year = DEFAULT_YEAR[r] || 2021
    setStartDate(`${year}-01-01`)
    setEndDate(`${year}-12-31`)
  }

  function handleSelectParcel(feature) {
    setSelected(feature)
    predictMutation.reset()
  }

  function handleRunPrediction() {
    if (!selected) return
    predictMutation.mutate({
      aoi: selected.geometry,
      startDate,
      endDate,
    })
  }

  const result = predictMutation.data
  const trueLabel = selected?.properties?.classname
  const match = result && trueLabel ? result.predicted_class === trueLabel : null

  return (
    <div>
      <h2>Parcel Browser</h2>
      <p>
        Real field boundaries — most pulled directly from the training fetch
        checkpoints (EuroCrops shapefiles + AgriFieldNet India, same data the
        model was trained and validated on), plus independent hand-labeled
        test sets that were never part of training. Pick a region, click a
        parcel outline, then run it through the live model and compare
        against its true label.
      </p>

      <fieldset>
        <legend>Region</legend>
        <label htmlFor="region-select">Region:</label>
        <select id="region-select" value={region} onChange={handleRegionChange}>
          {(regionsQuery.data || []).map((r) => (
            <option key={r.code} value={r.code}>
              {r.name} ({r.count} parcels available)
            </option>
          ))}
        </select>
        {parcelsQuery.isFetching && <span> loading parcels...</span>}
      </fieldset>

      <div style={{ display: 'flex', gap: 12 }}>
        <div style={{ flex: '1 1 65%', border: '1px solid #999999' }}>
          <MapContainer center={[20, 10]} zoom={3} style={{ height: 520, width: '100%' }}>
            <BaseLayers />
            {parcelsQuery.data && parcelsQuery.data.features.length > 0 && (
              <>
                <GeoJSON
                  key={region}
                  data={parcelsQuery.data}
                  style={(feature) => parcelStyle(feature, selected?.properties?.id)}
                  onEachFeature={(feature, layer) => {
                    layer.on('click', () => handleSelectParcel(feature))
                    layer.bindTooltip(feature.properties.classname)
                  }}
                  ref={geoJsonRef}
                />
                <FitBounds geojson={parcelsQuery.data} />
              </>
            )}
            <ZoomToSelected feature={selected} />
          </MapContainer>
        </div>

        <div style={{ flex: '1 1 35%' }}>
          <fieldset>
            <legend>Parcels in this region (sample of {parcelsQuery.data?.features.length ?? 0})</legend>
            <p style={{ fontSize: 11, color: '#555555', margin: '0 0 4px' }}>
              Real fields are spread across the whole country, not clustered together
              — pick one from this list to zoom the map to it (or zoom in yourself and
              click a boundary directly).
            </p>
            <div style={{ maxHeight: 140, overflowY: 'auto', border: '1px solid #cccccc' }}>
              <table>
                <tbody>
                  {(parcelsQuery.data?.features || []).map((f) => (
                    <tr
                      key={f.properties.id}
                      onClick={() => handleSelectParcel(f)}
                      style={{
                        cursor: 'pointer',
                        background:
                          selected?.properties?.id === f.properties.id ? '#c5d4e8' : undefined,
                      }}
                    >
                      <td>{f.properties.id}</td>
                      <td>{f.properties.classname}</td>
                      <td>{f.properties.area_hectares.toFixed(2)} ha</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </fieldset>

          <fieldset>
            <legend>Selected parcel</legend>
            {!selected && <p>Pick a parcel from the list above, or click a field outline on the map.</p>}
            {selected && (
              <table>
                <tbody>
                  <tr>
                    <th>ID</th>
                    <td>{selected.properties.id}</td>
                  </tr>
                  <tr>
                    <th>Region</th>
                    <td>{selected.properties.region_name}</td>
                  </tr>
                  <tr>
                    <th>True label</th>
                    <td>
                      <strong>{selected.properties.classname}</strong>
                    </td>
                  </tr>
                  <tr>
                    <th>Area</th>
                    <td>{selected.properties.area_hectares.toFixed(2)} ha</td>
                  </tr>
                </tbody>
              </table>
            )}
          </fieldset>

          {selected && (
            <fieldset>
              <legend>Run prediction</legend>
              <div>
                <label htmlFor="start-date">Start:</label>
                <input
                  id="start-date"
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              <div style={{ marginTop: 4 }}>
                <label htmlFor="end-date">End:</label>
                <input
                  id="end-date"
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
              <div style={{ marginTop: 8 }}>
                <button onClick={handleRunPrediction} disabled={predictMutation.isPending}>
                  {predictMutation.isPending ? 'Running...' : 'Run Prediction'}
                </button>
              </div>

              {predictMutation.isError && (
                <div className="status-banner error" style={{ marginTop: 8 }}>
                  {String(predictMutation.error.message)}
                </div>
              )}

              {result && (
                <div
                  className={`status-banner ${match ? 'ok' : 'error'}`}
                  style={{ marginTop: 8 }}
                >
                  Predicted: <strong>{result.predicted_class}</strong> (
                  {(result.confidence * 100).toFixed(1)}% confidence) — true label:{' '}
                  <strong>{trueLabel}</strong> — {match ? 'MATCH' : 'MISMATCH'}
                  <br />
                  area: {result.area_hectares.toFixed(2)} ha
                  <br />
                  observations used: {result.observations_used}
                </div>
              )}

              {result?.partial_range_warning && (
                <div className="status-banner" style={{ marginTop: 8 }}>
                  ⚠ {result.partial_range_warning}
                </div>
              )}

              {result && (
                <table style={{ marginTop: 8 }}>
                  <thead>
                    <tr>
                      <th>Class</th>
                      <th>Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([cls, prob]) => (
                        <tr key={cls}>
                          <td>{cls}</td>
                          <td>{(prob * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              )}
            </fieldset>
          )}
        </div>
      </div>
    </div>
  )
}
