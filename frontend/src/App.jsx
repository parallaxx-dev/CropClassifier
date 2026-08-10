import { NavLink, Outlet } from 'react-router-dom'
import './App.css'

function App() {
  return (
    <>
      <header className="site-header">
        <h1>Sentinel-2 Crop Classifier</h1>
        <div className="subtitle">
          multi-country + India crop type mapping — Sentinel-2 time-series transformer
        </div>
      </header>

      <nav className="site-nav">
        <ul>
          <li>
            <NavLink to="/parcels" className={({ isActive }) => (isActive ? 'active' : '')}>
              Parcel Browser
            </NavLink>
          </li>
          <li>
            <NavLink to="/draw" className={({ isActive }) => (isActive ? 'active' : '')}>
              Draw AOI
            </NavLink>
          </li>
          <li>
            <NavLink to="/stats" className={({ isActive }) => (isActive ? 'active' : '')}>
              Statistics
            </NavLink>
          </li>
        </ul>
      </nav>

      <main className="site-main">
        <Outlet />
      </main>

      <footer className="site-footer">
        Crop Classifier project — backend: FastAPI + PyTorch time-series transformer,
        data: EuroCrops + AgriFieldNet (India), fetched live via Copernicus Data Space
        Ecosystem. See progress.md / PROJECT_WALKTHROUGH.md in the repo for full details.
      </footer>
    </>
  )
}

export default App
