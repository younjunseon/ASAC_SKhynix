import { useState, useEffect } from 'react'
import Papa from 'papaparse'
import { dataUrl } from '../utils/dataUrl'

export function useCSV(path) {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!path) { setData([]); setLoading(false); return }
    setLoading(true)
    fetch(dataUrl(path))
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.text()
      })
      .then(text => {
        const result = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true })
        setData(result.data)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [path])

  return { data, loading }
}
