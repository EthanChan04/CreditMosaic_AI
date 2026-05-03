"use client";

import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import { ChartSkeleton } from "../shared/Skeleton";
import { EmptyState } from "../shared/EmptyState";

interface EChartsWrapperProps {
  option: echarts.EChartsOption;
  height?: number | string;
  loading?: boolean;
  empty?: boolean;
  emptyMessage?: string;
  className?: string;
  theme?: "light" | "dark";
}

export function EChartsWrapper({
  option,
  height = 300,
  loading = false,
  empty = false,
  emptyMessage = "No data available",
  className = "",
  theme,
}: EChartsWrapperProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!containerRef.current || loading || empty) return;

    const isDark = theme === "dark" ||
      (typeof document !== "undefined" &&
        document.documentElement.classList.contains("dark"));

    const chart = echarts.init(containerRef.current, isDark ? "dark" : undefined, {
      renderer: "canvas",
    });

    chartRef.current = chart;
    chart.setOption({ ...option, backgroundColor: "transparent" }, true);

    const handleResize = () => chart.resize();
    const observer = new ResizeObserver(handleResize);
    observer.observe(containerRef.current);

    return () => {
      observer.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, [option, loading, empty, theme]);

  if (loading) {
    return <ChartSkeleton />;
  }

  if (empty) {
    return (
      <div className={className} style={{ height }}>
        <EmptyState icon="📊" title={emptyMessage} />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`chart-container ${className}`}
      style={{ height, minHeight: 200 }}
    />
  );
}
