import { LayersControl, TileLayer } from 'react-leaflet'

const { BaseLayer } = LayersControl

// Same map-type categories Google Maps offers (Roadmap / Satellite / Terrain /
// hybrid labels-over-satellite), built from free tile providers that need no
// API key -- there's no Google Maps tile access without a billed API key, so
// these are the honest equivalent, not a reskin pretending to be Google's own
// tiles.
export default function BaseLayers() {
  return (
    <LayersControl position="topright">
      <BaseLayer checked name="Street (OpenStreetMap)">
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
      </BaseLayer>

      <BaseLayer name="Satellite (Esri World Imagery)">
        <TileLayer
          attribution="Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        />
      </BaseLayer>

      <BaseLayer name="Terrain (OpenTopoMap)">
        <TileLayer
          attribution='Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap (CC-BY-SA)'
          url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
        />
      </BaseLayer>

      <BaseLayer name="Light (CartoDB Positron)">
        <TileLayer
          attribution='&copy; OpenStreetMap contributors &copy; CARTO'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        />
      </BaseLayer>

      <LayersControl.Overlay name="Labels (over satellite)">
        <TileLayer
          attribution="Tiles &copy; Esri"
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
        />
      </LayersControl.Overlay>
    </LayersControl>
  )
}
