import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {title:"FuelWise 油耗領航員",description:"商用車行程油耗分析"};
export default function Layout({children}:{children:React.ReactNode}) {
  return <html lang="zh-Hant"><body>{children}</body></html>;
}
