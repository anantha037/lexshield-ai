import { ChatMessage } from "@/types";
import IntentBadge from "./IntentBadge";
import { formatTimestamp } from "@/lib/utils";
import { Scale, User } from "lucide-react";

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {/* Avatar */}
      <div
        className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? "bg-amber-500" : "bg-slate-700"
        }`}
      >
        {isUser ? (
          <User className="h-4 w-4 text-slate-950" />
        ) : (
          <Scale className="h-4 w-4 text-amber-400" />
        )}
      </div>

      {/* Bubble */}
      <div
        className={`max-w-[75%] space-y-1 flex flex-col ${
          isUser ? "items-end" : "items-start"
        }`}
      >
        {!isUser && message.intent && (
          <IntentBadge intent={message.intent} />
        )}

        <div
          className={`px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? "bg-amber-500 text-slate-950 font-medium rounded-tr-sm"
              : "bg-slate-800 text-slate-100 rounded-tl-sm"
          }`}
        >
          {message.content}
        </div>

        <span className="text-slate-600 text-xs">
          {formatTimestamp(message.timestamp)}
        </span>

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="space-y-1 w-full mt-1">
            {message.citations.slice(0, 2).map((c) => (
              <div
                key={c.chunk_id}
                className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs"
              >
                <span className="text-amber-400 font-medium">{c.source}</span>
                <span className="text-slate-500 mx-1">·</span>
                <span className="text-slate-400">{c.section}</span>
                <p className="text-slate-500 mt-1 line-clamp-2">{c.text}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}