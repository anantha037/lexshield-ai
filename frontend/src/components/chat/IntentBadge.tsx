import { IntentType } from "@/types";
import { cn } from "@/lib/utils";

const intentConfig: Record<IntentType, { label: string; className: string }> = {
  legal_query: {
    label: "Legal Query",
    className: "bg-blue-900/50 text-blue-300 border-blue-700",
  },
  risk_check: {
    label: "Risk Check",
    className: "bg-red-900/50 text-red-300 border-red-700",
  },
  document_draft: {
    label: "Drafting",
    className: "bg-purple-900/50 text-purple-300 border-purple-700",
  },
  ipc_lookup: {
    label: "IPC Lookup",
    className: "bg-orange-900/50 text-orange-300 border-orange-700",
  },
  case_search: {
    label: "Case Search",
    className: "bg-green-900/50 text-green-300 border-green-700",
  },
  general: {
    label: "General",
    className: "bg-slate-800 text-slate-400 border-slate-700",
  },
};

export default function IntentBadge({ intent }: { intent: IntentType }) {
  const config = intentConfig[intent] ?? intentConfig.general;
  return (
    <span
      className={cn(
        "text-xs px-2 py-0.5 rounded-full border font-medium",
        config.className
      )}
    >
      {config.label}
    </span>
  );
}