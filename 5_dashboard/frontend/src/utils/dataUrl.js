const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api/data`
  : ''

export const dataUrl = (path) => `${API_BASE}${path}`
