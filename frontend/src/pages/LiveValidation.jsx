import { useEffect, useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { MapContainer, GeoJSON, useMap } from 'react-leaflet'
import L from 'leaflet'
import { fetchDemoParcels, repredictDemoParcels } from '../api'
import BaseLayers from '../components/BaseLayers'

function FitBounds({ geojson }) {
  const map = useMap()
  useEffect(() => {
    if (!geojson || geojson.features.length === 0) return
    const bounds = L.geoJSON(geojson).getBounds()
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [20, 20] })
  }, [geojson, map])
  return null
}

function ZoomToSelected({ feature }) {
  const map = useMap()
  useEffect(() => {
    if (!feature) return
    const bounds = L.geoJSON(feature).getBounds()
    if (bounds.isValid()) map.flyToBounds(bounds, { padding: [80, 80], maxZoom: 17, duration: 0.5 })
  }, [feature, map])
  return null
}

// Fill = true label's color (same palette as Parcel Browser). Border encodes
// whether the cached prediction agrees with it -- green agrees, red
// disagrees, gray dashed means it hasn't been predicted yet this session.
function parcelStyle(feature, selectedId) {
  const { match, color } = feature.properties
  const isSelected = feature.properties.id === selectedId
  let borderColor = '#999999'
  let dashArray = '4 3'
  if (match === true) {
    borderColor = '#2e7d32'
    dashArray = null
  } else if (match === false) {
    borderColor = '#c62828'
    dashArray = null
  }
  return {
    color: isSelected ? '#000000' : borderColor,
    weight: isSelected ? 4 : match === null ? 2 : 3,
    dashArray: isSelected ? null : dashArray,
    fillColor: color,
    fillOpacity: isSelected ? 0.4 : 0.25,
  }
}

export default function LiveValidation() {
  const [selected, setSelected] = useState(null)
  const queryClient = useQueryClient()

  const parcelsQuery = useQuery({ queryKey: ['demo-parcels'], queryFn: fetchDemoParcels })

  const repredictMutation = useMutation({
    mutationFn: repredictDemoParcels,
    onSuccess: (data) => {
      queryClient.setQueryData(['demo-parcels'], data)
    },
  })

  const geojson = parcelsQuery.data
  const features = geojson?.features || []
  const predicted = features.filter((f) => f.properties.predicted_class !== null)
  const correct = predicted.filter((f) => f.properties.match).length
  const errored = features.filter((f) => f.properties.error).length

  return (
    <div>
      <h2>Live Validation</h2>
      <p>
        A small, spatially-clustered sample of real parcels — a few neighbouring fields from
        each region the model was actually trained on, not a random scatter across the whole
        country. Every parcel shows its <strong>true label</strong> (from the training/hand-labeled
        data) next to whatever the model most recently predicted for it. Predictions shown here
        are cached from the last run — click the button below to re-run the model live against
        Sentinel-2 right now and confirm these aren't hardcoded.
      </p>

      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <button onClick={() => repredictMutation.mutate()} disabled={repredictMutation.isPending}>
          {repredictMutation.isPending
            ? `Re-predicting all ${features.length} parcels live... (~1-2 min)`
            : `Re-predict all ${features.length} parcels`}
        </button>
        {predicted.length > 0 && (
          <span className="status-banner ok" style={{ margin: 0 }}>
            {correct} / {predicted.length} correct{errored > 0 ? ` (${errored} errored)` : ''}
          </span>
        )}
      </div>

      {repredictMutation.isError && (
        <div className="status-banner error" style={{ marginBottom: 8 }}>
          {String(repredictMutation.error.message)}
        </div>
      )}

      {parcelsQuery.isLoading && <p>loading curated parcels...</p>}
      {parcelsQuery.isError && (
        <div className="status-banner error">{String(parcelsQuery.error.message)}</div>
      )}

      {geojson && (
        <div style={{ display: 'flex', gap: 12 }}>
          <div style={{ flex: '1 1 60%', border: '1px solid #999999' }}>
            <MapContainer center={[20, 10]} zoom={3} style={{ height: 560, width: '100%' }}>
              <BaseLayers />
              <GeoJSON
                key={JSON.stringify(features.map((f) => f.properties.predicted_class))}
                data={geojson}
                style={(feature) => parcelStyle(feature, selected?.properties?.id)}
                onEachFeature={(feature, layer) => {
                  layer.on('click', () => setSelected(feature))
                  const p = feature.properties
                  const predText = p.error
                    ? `error: ${p.error}`
                    : p.predicted_class
                      ? `${p.predicted_class} (${(p.confidence * 100).toFixed(0)}%)`
                      : 'not yet predicted'
                  layer.bindTooltip(`${p.region_name}: true=${p.classname}, predicted=${predText}`)
                }}
              />
              <FitBounds geojson={geojson} />
              <ZoomToSelected feature={selected} />
            </MapContainer>
          </div>

          <div style={{ flex: '1 1 40%' }}>
            <fieldset>
              <legend>All {features.length} parcels (click a row or a field outline)</legend>
              <div style={{ maxHeight: 520, overflowY: 'auto' }}>
                <table>
                  <thead>
                    <tr>
                      <th>Region</th>
                      <th>True</th>
                      <th>Area</th>
                      <th>Predicted</th>
                      <th>Conf.</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {features.map((f) => {
                      const p = f.properties
                      return (
                        <tr
                          key={p.id}
                          onClick={() => setSelected(f)}
                          style={{
                            cursor: 'pointer',
                            background: selected?.properties?.id === p.id ? '#c5d4e8' : undefined,
                          }}
                        >
                          <td>{p.region_name}</td>
                          <td>{p.classname}</td>
                          <td>{p.area_hectares.toFixed(2)} ha</td>
                          <td>
                            {p.error ? (
                              <span style={{ color: '#c62828', fontSize: 11 }}>error</span>
                            ) : (
                              p.predicted_class || <span style={{ color: '#999999' }}>—</span>
                            )}
                          </td>
                          <td>{p.confidence != null ? `${(p.confidence * 100).toFixed(0)}%` : ''}</td>
                          <td>
                            {p.match === true && <span style={{ color: '#2e7d32' }}>MATCH</span>}
                            {p.match === false && <span style={{ color: '#c62828' }}>MISS</span>}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </fieldset>

            {selected && (
              <fieldset style={{ marginTop: 8 }}>
                <legend>Selected parcel</legend>
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
                    <tr>
                      <th>Predicted</th>
                      <td>
                        {selected.properties.error
                          ? `error: ${selected.properties.error}`
                          : selected.properties.predicted_class || 'not yet predicted'}
                      </td>
                    </tr>
                    <tr>
                      <th>Confidence</th>
                      <td>
                        {selected.properties.confidence != null
                          ? `${(selected.properties.confidence * 100).toFixed(1)}%`
                          : ''}
                      </td>
                    </tr>
                    <tr>
                      <th>Observations used</th>
                      <td>{selected.properties.observations_used ?? ''}</td>
                    </tr>
                  </tbody>
                </table>
              </fieldset>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
