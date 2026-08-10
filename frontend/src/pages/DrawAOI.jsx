import { useEffect, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { MapContainer, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet-draw'
import { predict, uploadAoiFile } from '../api'
import BaseLayers from '../components/BaseLayers'
import { parsePolygonInput, approxAreaHectares } from '../utils/geoParse'

// react-leaflet has no built-in wrapper for the leaflet-draw plugin -- it's
// an old-style Leaflet plugin that extends the global L object, not an ES
// module of its own. This wires it up directly against the map instance.
//
// externalGeometry: a GeoJSON Polygon set from OUTSIDE the draw tool (pasted
// coordinates, an uploaded file) -- pushed into the SAME drawnItems
// FeatureGroup the draw tool itself uses, so there's one visual source of
// truth (edit/delete toolbar buttons act on it either way) instead of a
// second overlapping layer to keep in sync.
function DrawControl({ onDrawn, externalGeometry }) {
  const map = useMap()
  const drawnItemsRef = useRef(null)

  useEffect(() => {
    const drawnItems = new L.FeatureGroup()
    drawnItemsRef.current = drawnItems
    map.addLayer(drawnItems)

    const drawControl = new L.Control.Draw({
      draw: {
        // showArea: false -- leaflet-draw@1.0.4 (last released 2021, unmaintained)
        // has a real bug in its bundled GeometryUtil.readableArea() that throws
        // "ReferenceError: type is not defined" on every vertex add / mousemove
        // when the live area tooltip is on. That exception fires deep inside
        // Leaflet's own event dispatch, mid-vertex-add, and corrupts the
        // in-progress polygon's internal state -- this is what caused "can't
        // add more than ~3 vertices": every new vertex's tooltip update crashed.
        // The area tooltip itself is a nice-to-have, not load-bearing (area is
        // shown after submission anyway), so disabling it is the direct fix
        // rather than patching a bundled, abandoned dependency.
        polygon: { allowIntersection: false, showArea: false },
        rectangle: { showArea: false },
        polyline: false,
        circle: false,
        circlemarker: false,
        marker: false,
      },
      edit: {
        featureGroup: drawnItems,
      },
    })
    map.addControl(drawControl)

    function handleCreated(e) {
      // one AOI at a time -- clear whatever was drawn before
      drawnItems.clearLayers()
      drawnItems.addLayer(e.layer)
      onDrawn(e.layer.toGeoJSON().geometry)
    }
    function handleDeleted() {
      onDrawn(null)
    }

    map.on(L.Draw.Event.CREATED, handleCreated)
    map.on(L.Draw.Event.DELETED, handleDeleted)

    return () => {
      map.off(L.Draw.Event.CREATED, handleCreated)
      map.off(L.Draw.Event.DELETED, handleDeleted)
      map.removeControl(drawControl)
      map.removeLayer(drawnItems)
    }
  }, [map, onDrawn])

  useEffect(() => {
    if (!externalGeometry || !drawnItemsRef.current) return
    drawnItemsRef.current.clearLayers()
    const layer = L.geoJSON(externalGeometry)
    drawnItemsRef.current.addLayer(layer)
    onDrawn(externalGeometry)
    const bounds = layer.getBounds()
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [40, 40] })
  }, [externalGeometry, map, onDrawn])

  return null
}

export default function DrawAOI() {
  const [aoi, setAoi] = useState(null)
  const [startDate, setStartDate] = useState('2021-01-01')
  const [endDate, setEndDate] = useState('2021-12-31')
  const [externalGeometry, setExternalGeometry] = useState(null)
  const [pastedText, setPastedText] = useState('')
  const [pasteError, setPasteError] = useState(null)

  const predictMutation = useMutation({ mutationFn: predict })
  const uploadMutation = useMutation({
    mutationFn: uploadAoiFile,
    onSuccess: (data) => setExternalGeometry(data.aoi),
  })

  function handleSubmit() {
    if (!aoi) return
    predictMutation.mutate({ aoi, startDate, endDate })
  }

  function handleUseCoordinates() {
    const parsed = parsePolygonInput(pastedText)
    if (!parsed) {
      setPasteError(
        'Could not parse that -- expected GeoJSON (a Polygon geometry or Feature), WKT ' +
          '(POLYGON((lon lat, lon lat, ...))), or one "lon,lat" pair per line (at least 3 points).',
      )
      return
    }
    setPasteError(null)
    setExternalGeometry(parsed)
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0]
    if (file) uploadMutation.mutate(file)
    e.target.value = '' // allow re-uploading the same filename after a fix
  }

  const result = predictMutation.data

  return (
    <div>
      <h2>Draw an Area of Interest</h2>
      <p>
        Use the polygon or rectangle tool (top-left of the map) to outline a field
        boundary anywhere. Field size must be between 0.1 and 500 hectares — the
        backend fetches its real Sentinel-2 growing-season time series live from
        the Copernicus Data Space Ecosystem for whatever you draw.
      </p>

      <div style={{ display: 'flex', gap: 12 }}>
        <div style={{ flex: '1 1 65%', border: '1px solid #999999' }}>
          <MapContainer center={[48.2, 15.6]} zoom={13} style={{ height: 520, width: '100%' }}>
            <BaseLayers />
            <DrawControl onDrawn={setAoi} externalGeometry={externalGeometry} />
          </MapContainer>
        </div>

        <div style={{ flex: '1 1 35%' }}>
          <fieldset>
            <legend>Or paste coordinates</legend>
            <p style={{ fontSize: 11, color: '#555555', margin: '0 0 4px' }}>
              GeoJSON, WKT (<code>POLYGON((lon lat, ...))</code>), or one <code>lon,lat</code> pair per
              line (at least 3 points).
            </p>
            <textarea
              rows={4}
              style={{ width: '100%', fontSize: 11, fontFamily: 'monospace' }}
              value={pastedText}
              onChange={(e) => setPastedText(e.target.value)}
              placeholder={'82.9375,25.9658\n82.9382,25.9657\n82.9380,25.9649'}
            />
            <div style={{ marginTop: 4 }}>
              <button onClick={handleUseCoordinates} disabled={!pastedText.trim()}>
                Use these coordinates
              </button>
            </div>
            {pasteError && (
              <div className="status-banner error" style={{ marginTop: 8 }}>
                {pasteError}
              </div>
            )}
          </fieldset>

          <fieldset>
            <legend>Or upload a file</legend>
            <p style={{ fontSize: 11, color: '#555555', margin: '0 0 4px' }}>
              KML, GeoPackage (.gpkg), GeoJSON, zipped Shapefile (.zip), GeoTIFF, or most other
              GDAL-readable vector/raster formats. Vector files use the largest polygon found;
              rasters use their georeferenced footprint.
            </p>
            <input
              type="file"
              accept=".kml,.gpkg,.geojson,.json,.zip,.tif,.tiff,.shp"
              onChange={handleFileChange}
              disabled={uploadMutation.isPending}
            />
            {uploadMutation.isPending && <p style={{ fontSize: 11 }}>parsing file...</p>}
            {uploadMutation.isError && (
              <div className="status-banner error" style={{ marginTop: 8 }}>
                {String(uploadMutation.error.message)}
              </div>
            )}
            {uploadMutation.isSuccess && (
              <div className="status-banner ok" style={{ marginTop: 8 }}>
                Loaded AOI from {uploadMutation.data.source_filename} (
                {uploadMutation.data.area_hectares.toFixed(2)} ha)
              </div>
            )}
          </fieldset>

          <fieldset>
            <legend>Date range</legend>
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
          </fieldset>

          <fieldset>
            <legend>Prediction</legend>
            {!aoi && <p>Draw a polygon or rectangle on the map first.</p>}
            {aoi && (
              <p style={{ fontSize: 12 }}>
                AOI area: <strong>{approxAreaHectares(aoi).toFixed(2)} ha</strong>
              </p>
            )}
            {aoi && (
              <button onClick={handleSubmit} disabled={predictMutation.isPending}>
                {predictMutation.isPending ? 'Fetching Sentinel-2 + running model...' : 'Run Prediction'}
              </button>
            )}

            {predictMutation.isError && (
              <div className="status-banner error" style={{ marginTop: 8 }}>
                {String(predictMutation.error.message)}
              </div>
            )}

            {result && (
              <>
                <div className="status-banner ok" style={{ marginTop: 8 }}>
                  Predicted: <strong>{result.predicted_class}</strong> (
                  {(result.confidence * 100).toFixed(1)}% confidence)
                  <br />
                  area: {result.area_hectares.toFixed(2)} ha
                  <br />
                  observations used: {result.observations_used} / 45
                </div>
                {result.partial_range_warning && (
                  <div className="status-banner" style={{ marginTop: 8 }}>
                    ⚠ {result.partial_range_warning}
                  </div>
                )}
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
              </>
            )}
          </fieldset>
        </div>
      </div>
    </div>
  )
}
