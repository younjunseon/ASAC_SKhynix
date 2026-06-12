import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Drilldown from "./pages/Drilldown";
import DrilldownWafer from "./pages/DrilldownWafer";
import DrilldownLot from "./pages/DrilldownLot";
import Model from "./pages/Model";
import Data from "./pages/Data";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="data" element={<Data />} />
          {/* 다이 레벨 정밀 분석(드릴다운) — 하위: 다이 / 웨이퍼 / 로트 차원 */}
          <Route path="drilldown" element={<Drilldown />} />
          <Route path="drilldown/wafer" element={<DrilldownWafer />} />
          <Route path="drilldown/lot" element={<DrilldownLot />} />
          <Route path="model" element={<Model />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
