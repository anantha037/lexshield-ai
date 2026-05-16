"use client";
import { useEffect, useState } from "react";

interface RiskGaugeProps {
  score: number; // 0.0 to 1.0
}

function getRiskLabel(score: number) {
  if (score < 0.3) return { label: "Low Risk", color: "#22c55e" };
  if (score < 0.7) return { label: "Moderate Risk", color: "#f59e0b" };
  return { label: "High Risk", color: "#ef4444" };
}

export default function RiskGauge({ score }: RiskGaugeProps) {
  const [animated, setAnimated] = useState(0);
  const { label, color } = getRiskLabel(score);

  useEffect(() => {
    const timer = setTimeout(() => setAnimated(score), 100);
    return () => clearTimeout(timer);
  }, [score]);

  const radius = 70;
  const cx = 100;
  const cy = 100;
  const startAngle = -210;
  const endAngle = 30;
  const totalAngle = endAngle - startAngle;
  const currentAngle = startAngle + totalAngle * animated;

  const toRad = (deg: number) => (deg * Math.PI) / 180;

  const arcPath = (start: number, end: number) => {
    const s = {
      x: cx + radius * Math.cos(toRad(start)),
      y: cy + radius * Math.sin(toRad(start)),
    };
    const e = {
      x: cx + radius * Math.cos(toRad(end)),
      y: cy + radius * Math.sin(toRad(end)),
    };
    const large = end - start > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${radius} ${radius} 0 ${large} 1 ${e.x} ${e.y}`;
  };

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 140" className="w-48 h-32">
        {/* Background arc */}
        <path
          d={arcPath(startAngle, endAngle)}
          fill="none"
          stroke="#1e293b"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Colored arc */}
        <path
          d={arcPath(startAngle, currentAngle)}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          style={{ transition: "all 1s ease-out" }}
        />
        {/* Score */}
        <text
          x="100"
          y="95"
          textAnchor="middle"
          fill="white"
          fontSize="22"
          fontWeight="bold"
        >
          {(score * 100).toFixed(0)}
        </text>
        <text x="100" y="112" textAnchor="middle" fill="#64748b" fontSize="10">
          out of 100
        </text>
      </svg>
      <span
        className="text-sm font-semibold px-3 py-1 rounded-full"
        style={{ backgroundColor: `${color}20`, color }}
      >
        {label}
      </span>
    </div>
  );
}