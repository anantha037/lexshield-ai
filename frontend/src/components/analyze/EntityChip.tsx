import { NEREntity } from "@/types";
import { cn } from "@/lib/utils";

const entityConfig: Record<NEREntity["label"], { className: string }> = {
  IPC_SECTION: { className: "bg-red-900/40 text-red-300 border-red-700" },
  ACT:         { className: "bg-blue-900/40 text-blue-300 border-blue-700" },
  DATE:        { className: "bg-green-900/40 text-green-300 border-green-700" },
  PARTY:       { className: "bg-purple-900/40 text-purple-300 border-purple-700" },
  COURT:       { className: "bg-orange-900/40 text-orange-300 border-orange-700" },
  OTHER:       { className: "bg-slate-800 text-slate-400 border-slate-700" },
};

export default function EntityChip({ entity }: { entity: NEREntity }) {
  const config = entityConfig[entity.label] ?? entityConfig.OTHER;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border",
        config.className
      )}
    >
      <span className="opacity-60 text-[10px]">{entity.label}</span>
      <span className="font-medium">{entity.text}</span>
    </span>
  );
}