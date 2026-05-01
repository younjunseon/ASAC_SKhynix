import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Drilldown from "./pages/Drilldown";
import Model from "./pages/Model";
import Data from "./pages/Data";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="drilldown" element={<Drilldown />} />
          <Route path="model" element={<Model />} />
          <Route path="data" element={<Data />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}