import { Citation } from "@/types";
import { BookOpen } from "lucide-react";

export default function CitationCard({
  citation,
}: {
  citation: Citation;
  index: number;
}) {
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-xl p-4 space-y-2">
      <div className="flex items-start gap-2">
        <div className="bg-amber-500/10 p-1.5 rounded-lg mt-0.5">
          <BookOpen className="h-3.5 w-3.5 text-amber-400" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-amber-400 text-xs font-semibold">
              {citation.source}
            </span>
            <span className="text-slate-600 text-xs">·</span>
            <span className="text-slate-400 text-xs">{citation.section}</span>
            <span className="ml-auto text-slate-600 text-xs">
              Score: {(citation.score * 100).toFixed(0)}%
            </span>
          </div>
          <p className="text-slate-400 text-xs mt-1.5 leading-relaxed line-clamp-3">
            {citation.text}
          </p>
        </div>
      </div>
    </div>
  );
}