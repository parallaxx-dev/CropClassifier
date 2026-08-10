import { useQuery } from '@tanstack/react-query'
import { fetchModelInfo } from '../api'
import Heatmap, { sequentialBlue, divergingBlueRed, ScaleLegend } from '../components/Heatmap'

function pct(x) {
  return `${(x * 100).toFixed(1)}%`
}

export default function Statistics() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['model-info'],
    queryFn: fetchModelInfo,
  })

  if (isLoading) return <p>loading model statistics...</p>
  if (isError) return <div className="status-banner error">{String(error.message)}</div>
  if (!data) return null

  return (
    <div>
      <h2>Model &amp; Pipeline Statistics</h2>
      <p>
        Precomputed from the actual training run and live-pipeline testing of the
        currently deployed checkpoint (<code>{data.checkpoint}</code>, {data.num_classes}{' '}
        classes). Not live-recomputed per page load — see progress.md in the repo
        for the full methodology.
      </p>

      <h3>Training data</h3>
      <table>
        <thead>
          <tr>
            <th>Region</th>
            <th>Parcels</th>
            <th>Total area</th>
            <th>Mean parcel size</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data.training_data).map(([code, d]) => (
            <tr key={code}>
              <td>
                {d.name} ({code})
              </td>
              <td>{d.parcels.toLocaleString()}</td>
              <td>{d.total_area_hectares.toLocaleString(undefined, { maximumFractionDigits: 1 })} ha</td>
              <td>{(d.total_area_hectares / d.parcels).toFixed(2)} ha</td>
              <td>{d.status}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Overall model accuracy</h3>
      <table>
        <tbody>
          <tr>
            <th>Single-draw accuracy</th>
            <td>{pct(data.overall.single_draw_accuracy)}</td>
          </tr>
          <tr>
            <th>Ensembled accuracy (production)</th>
            <td>{pct(data.overall.ensembled_accuracy)}</td>
          </tr>
          <tr>
            <th>Macro F1</th>
            <td>{data.overall.macro_f1.toFixed(2)}</td>
          </tr>
          <tr>
            <th>Weighted F1</th>
            <td>{data.overall.weighted_f1.toFixed(2)}</td>
          </tr>
          <tr>
            <th>Test set size</th>
            <td>{data.overall.n_test}</td>
          </tr>
        </tbody>
      </table>
      <p style={{ fontSize: 11, color: '#555555' }}>{data.overall.note}</p>

      <h3>Crop-wise (per-class)</h3>
      <table>
        <thead>
          <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Support</th>
          </tr>
        </thead>
        <tbody>
          {data.per_class.map((c) => (
            <tr key={c.class}>
              <td>{c.class}</td>
              <td>{c.precision.toFixed(2)}</td>
              <td>{c.recall.toFixed(2)}</td>
              <td>
                <strong>{c.f1.toFixed(2)}</strong>
              </td>
              <td>{c.support}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Region-wise (training-split evaluation)</h3>
      <table>
        <thead>
          <tr>
            <th>Region</th>
            <th>Accuracy</th>
            <th>n</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          {data.per_region_training_split.map((r) => (
            <tr key={r.region}>
              <td>{r.name}</td>
              <td>{pct(r.accuracy)}</td>
              <td>{r.n}</td>
              <td style={{ fontSize: 11, color: '#555555' }}>{r.note || ''}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Whole pipeline (live API test, real Sentinel-2 fetches)</h3>
      <p style={{ fontSize: 11, color: '#555555' }}>{data.live_pipeline.note}</p>
      <table>
        <thead>
          <tr>
            <th>Region</th>
            <th>Correct / n</th>
            <th>Accuracy</th>
            <th>Held out?</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          {data.live_pipeline.by_region.map((r) => (
            <tr key={r.region}>
              <td>{r.name}</td>
              <td>
                {r.correct} / {r.n}
              </td>
              <td>{pct(r.accuracy)}</td>
              <td>{r.held_out ? 'yes' : 'no'}</td>
              <td style={{ fontSize: 11, color: '#555555' }}>{r.note || ''}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>India, by class (genuinely held-out fields)</h4>
      <table>
        <thead>
          <tr>
            <th>Class</th>
            <th>Correct / n</th>
            <th>Accuracy</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          {data.live_pipeline.india_by_class.map((c) => (
            <tr key={c.class}>
              <td>{c.class}</td>
              <td>
                {c.correct} / {c.n}
              </td>
              <td>{pct(c.accuracy)}</td>
              <td style={{ fontSize: 11, color: '#555555' }}>{c.note || ''}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {data.confusion_matrix && (
        <>
          <h3>Confusion matrix</h3>
          <p style={{ fontSize: 11, color: '#555555' }}>{data.confusion_matrix.note}</p>
          <p style={{ fontSize: 11, color: '#555555' }}>
            Shaded by row (% of each true class's parcels landing in each predicted class) —
            darker means more of that true class's parcels landed there. Hover a cell for the
            raw count.
          </p>
          <ScaleLegend
            stops={[sequentialBlue(0), sequentialBlue(0.5), sequentialBlue(1)]}
            labels={['0% of row', '100% of row']}
          />
          <Heatmap
            labels={data.confusion_matrix.class_names}
            matrix={data.confusion_matrix.matrix}
            rowLabel="true"
            colLabel="predicted"
            cellColor={(v, i) => {
              const rowTotal = data.confusion_matrix.matrix[i].reduce((a, b) => a + b, 0)
              return sequentialBlue(rowTotal > 0 ? v / rowTotal : 0)
            }}
            cellText={(v) => (v > 0 ? v : '')}
          />
        </>
      )}

      {data.band_correlation && (
        <>
          <h3>Sentinel-2 band correlation</h3>
          <p style={{ fontSize: 11, color: '#555555' }}>{data.band_correlation.note}</p>
          <p style={{ fontSize: 11, color: '#555555' }}>
            Pearson correlation between the model's 13 input bands. Deep blue = strongly
            positive, deep red = strongly negative, pale = uncorrelated — the visible/NIR
            bands (B2-B8A) move almost together, while B10 (cirrus) is nearly independent
            of everything else.
          </p>
          <ScaleLegend
            stops={[divergingBlueRed(-1), divergingBlueRed(0), divergingBlueRed(1)]}
            labels={['-1.0', '+1.0']}
          />
          <Heatmap
            labels={data.band_correlation.bands}
            matrix={data.band_correlation.matrix}
            cellColor={(v) => divergingBlueRed(v)}
            cellText={(v) => v.toFixed(2)}
          />
        </>
      )}

      {data.feature_importance && (
        <>
          <h3>Feature importance (which Sentinel-2 bands drive predictions)</h3>
          <p style={{ fontSize: 11, color: '#555555' }}>{data.feature_importance.note}</p>
          <div style={{ maxWidth: 640 }}>
            {data.feature_importance.importances.map((f) => {
              const maxDrop = data.feature_importance.importances[0].accuracy_drop
              const pct = (f.accuracy_drop / maxDrop) * 100
              return (
                <div key={f.band} style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '3px 0' }}>
                  <div style={{ width: 32, fontSize: 11, color: '#52514e', textAlign: 'right' }}>{f.band}</div>
                  <div style={{ flex: 1, background: '#f0efec', height: 16 }}>
                    <div
                      style={{
                        width: `${pct}%`,
                        height: '100%',
                        background: '#2a78d6',
                        borderRadius: '0 4px 4px 0',
                      }}
                    />
                  </div>
                  <div style={{ width: 60, fontSize: 11, color: '#52514e' }}>
                    -{(f.accuracy_drop * 100).toFixed(1)}pp
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
