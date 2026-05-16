"use client";
import { useState, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { Upload, FileText, Loader2, AlertCircle, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import RiskGauge from "@/components/analyze/RiskGauge";
import EntityChip from "@/components/analyze/EntityChip";
import CitationCard from "@/components/analyze/CitationCard";
import { DocumentAnalysisResult } from "@/types";
import { analyzeDocument } from "@/lib/api-client";

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [result, setResult] = useState<DocumentAnalysisResult | null>(null);

  const { mutate, isPending, error, reset } = useMutation({
    mutationFn: (f: File) => {
      const fd = new FormData();
      fd.append("file", f);
      return analyzeDocument(fd);
    },
    onSuccess: (res) => {
        const d = res.data;
        setResult({
            document_type: d.document_type ?? d.doc_type ?? "Unknown",
            classification: d.classification ?? d.document_class ?? "",
            risk_score: {
            overall: d.risk_score?.overall ?? d.risk?.score ?? 0,
            breakdown: d.risk_score?.breakdown ?? [],
            },
            entities: d.entities ?? d.ner_entities ?? [],
            citations: d.citations ?? [],
            summary: d.summary ?? d.answer_text ?? "",
            raw_text: d.raw_text ?? "",
        });
    },

  });

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setFile(e.target.files[0]);
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    reset();
  };

  const apiError = error as any;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-white font-semibold">Document Deep-Dive</h1>
        <p className="text-slate-500 text-xs">
          OCR · NER · Risk Scoring · Citation Retrieval
        </p>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* LEFT — Upload Panel */}
        <div className="w-80 border-r border-slate-800 flex flex-col p-4 gap-4">
          <div
            onDrop={handleDrop}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            className={`border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer ${
              dragOver
                ? "border-amber-500 bg-amber-500/5"
                : "border-slate-700 hover:border-slate-600"
            }`}
            onClick={() => document.getElementById("file-input")?.click()}
          >
            <input
              id="file-input"
              type="file"
              accept=".pdf,.png,.jpg,.jpeg,.txt"
              className="hidden"
              onChange={handleFileChange}
            />
            <Upload className="h-8 w-8 text-slate-500 mx-auto mb-2" />
            <p className="text-slate-400 text-sm">
              Drop file here or click to browse
            </p>
            <p className="text-slate-600 text-xs mt-1">PDF · PNG · JPG · TXT</p>
          </div>

          {file && (
            <div className="bg-slate-800 rounded-lg px-3 py-2 flex items-center gap-2">
              <FileText className="h-4 w-4 text-amber-400 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-white text-xs font-medium truncate">
                  {file.name}
                </p>
                <p className="text-slate-500 text-xs">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
              </div>
              <button onClick={handleReset}>
                <X className="h-4 w-4 text-slate-500 hover:text-red-400" />
              </button>
            </div>
          )}

          <Button
            onClick={() => file && mutate(file)}
            disabled={!file || isPending}
            className="w-full bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold"
          >
            {isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              "Run Analysis"
            )}
          </Button>

          {isPending && (
            <div className="space-y-2 text-xs text-slate-500">
              <p>⟳ Running OCR...</p>
              <p>⟳ Extracting legal entities...</p>
              <p>⟳ Scoring risk clauses...</p>
              <p>⟳ Retrieving citations...</p>
            </div>
          )}

          {apiError && (
            <div className="flex items-start gap-2 bg-red-950/50 border border-red-800 rounded-lg p-3 text-red-300 text-xs">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              {apiError.message}
            </div>
          )}
        </div>

        {/* RIGHT — Results Panel */}
        <ScrollArea className="flex-1">
          {!result ? (
            <div className="flex items-center justify-center h-full text-slate-600 text-sm">
              Upload a document and run analysis to see results
            </div>
          ) : (
            <div className="p-6 space-y-6 max-w-2xl">
              {/* Doc Type */}
              <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
                <p className="text-slate-500 text-xs mb-1">Document Type</p>
                <p className="text-white font-semibold">{result.document_type}</p>
                <p className="text-amber-400 text-sm mt-1">
                  {result.classification}
                </p>
              </div>

              {/* Risk Gauge */}
              <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
                <p className="text-slate-400 text-sm font-medium mb-3">
                  Risk Assessment
                </p>
                <RiskGauge score={result.risk_score?.overall ?? result.risk?.score ?? 0} />
                {(result.risk_score?.breakdown?.length ?? 0) > 0 && (
                  <>
                    <Separator className="my-3 bg-slate-800" />
                    <div className="space-y-2">
                      {result.risk_score?.breakdown?.map((b) => (
                        <div
                          key={b.category}
                          className="flex items-center justify-between text-xs"
                        >
                          <span className="text-slate-400">{b.category}</span>
                          <div className="flex items-center gap-2">
                            <div className="w-24 bg-slate-800 rounded-full h-1.5">
                              <div
                                className="h-1.5 rounded-full bg-amber-500"
                                style={{ width: `${b.score * 100}%` }}
                              />
                            </div>
                            <span className="text-slate-500 w-8 text-right">
                              {(b.score * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>

              {/* NER Entities */}
              {result.entities.length > 0 && (
                <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
                  <p className="text-slate-400 text-sm font-medium mb-3">
                    Legal Entities ({result.entities.length})
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {result.entities.map((e, i) => (
                      <EntityChip key={i} entity={e} />
                    ))}
                  </div>
                </div>
              )}

              {/* Summary */}
              {result.summary && (
                <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
                  <p className="text-slate-400 text-sm font-medium mb-2">
                    Summary
                  </p>
                  <p className="text-slate-300 text-sm leading-relaxed">
                    {result.summary}
                  </p>
                </div>
              )}

              {/* Citations */}
              {result.citations.length > 0 && (
                <div className="space-y-2">
                  <p className="text-slate-400 text-sm font-medium">
                    Legal Citations ({result.citations.length})
                  </p>
                  {result.citations.map((c, i) => (
                    <CitationCard key={c.chunk_id} citation={c} index={i} />
                  ))}
                </div>
              )}
            </div>
          )}
        </ScrollArea>
      </div>
    </div>
  );
}