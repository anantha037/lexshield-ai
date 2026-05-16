"use client";
import { useState, useRef, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { Send, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import MessageBubble from "@/components/chat/MessageBubble";
import { ChatMessage } from "@/types";
import { chatQuery } from "@/lib/api-client";
import { generateSessionId } from "@/lib/utils";

const SESSION_ID = generateSessionId();

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Namaste! I'm LexShield AI. Ask me anything about Indian law — IPC sections, contract clauses, FIR analysis, or legal risk assessment.",
      intent: "general",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const { mutate, isPending, error } = useMutation({
    mutationFn: (query: string) =>
      chatQuery({ query, session_id: SESSION_ID, run_rag: true }),
    onSuccess: (res) => {
      const data = res.data;
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: data.answer_text ?? data.response ?? data.answer ?? "No response received.",
          intent: data.intent ?? "general",
          citations: data.citations ?? [],
          timestamp: new Date().toISOString(),
        },
      ]);
    },
  });

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isPending) return;

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        role: "user",
        content: trimmed,
        timestamp: new Date().toISOString(),
      },
    ]);
    setInput("");
    mutate(trimmed);
  };

  const apiError = error as any;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-white font-semibold">Chat Intelligence</h1>
        <p className="text-slate-500 text-xs">
          Powered by LLaMA 3.3 70B + ChromaDB RAG
        </p>
      </div>

      {/* Error Banner */}
      {apiError && (
        <div className="mx-6 mt-4 flex items-center gap-2 bg-red-950/50 border border-red-800 rounded-lg px-4 py-2.5 text-red-300 text-sm">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {apiError.message ?? "An error occurred."}
        </div>
      )}

      {/* Messages */}
      <ScrollArea className="flex-1 px-6 py-4">
        <div className="space-y-6 max-w-3xl mx-auto">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          {isPending && (
            <div className="flex items-center gap-2 text-slate-500 text-sm">
              <Loader2 className="h-4 w-4 animate-spin" />
              LexShield is analyzing your query...
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-slate-800 px-6 py-4">
        <div className="max-w-3xl mx-auto flex gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask about IPC sections, contract risks, case law..."
            rows={1}
            className="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-white placeholder:text-slate-500 text-sm resize-none focus:outline-none focus:border-amber-500 transition-colors"
          />
          <Button
            onClick={handleSend}
            disabled={isPending || !input.trim()}
            className="bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold px-4 self-end"
          >
            {isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
        <p className="text-center text-slate-700 text-xs mt-2">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}