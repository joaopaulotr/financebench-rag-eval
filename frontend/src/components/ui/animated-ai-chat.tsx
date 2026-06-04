"use client";

import { useEffect, useRef, useCallback } from "react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import {
    ArrowUpIcon,
    SendIcon,
    LoaderIcon,
    FileText,
    Sparkles,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import * as React from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface UseAutoResizeTextareaProps {
    minHeight: number;
    maxHeight?: number;
}

function useAutoResizeTextarea({
    minHeight,
    maxHeight,
}: UseAutoResizeTextareaProps) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const adjustHeight = useCallback(
        (reset?: boolean) => {
            const textarea = textareaRef.current;
            if (!textarea) return;

            if (reset) {
                textarea.style.height = `${minHeight}px`;
                return;
            }

            textarea.style.height = `${minHeight}px`;
            const newHeight = Math.max(
                minHeight,
                Math.min(
                    textarea.scrollHeight,
                    maxHeight ?? Number.POSITIVE_INFINITY
                )
            );

            textarea.style.height = `${newHeight}px`;
        },
        [minHeight, maxHeight]
    );

    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = `${minHeight}px`;
        }
    }, [minHeight]);

    useEffect(() => {
        const handleResize = () => adjustHeight();
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, [adjustHeight]);

    return { textareaRef, adjustHeight };
}

interface ExamplePrompt {
    label: string;
    query: string;
}

// Text-only example questions — clicking fills the input (no slash commands,
// no file/image attachments; this UI only sends text to the LangGraph backend).
const EXAMPLE_PROMPTS: ExamplePrompt[] = [
    {
        label: "3M FY2018 capex",
        query:
            "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Rely on the cash flow statement.",
    },
    {
        label: "Adobe FY2022 revenue",
        query: "What was Adobe's total revenue in FY2022 (in USD millions)?",
    },
    {
        label: "Amazon FY2021 operating margin",
        query: "What was Amazon's operating margin in FY2021?",
    },
];

interface TextareaProps
    extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
    containerClassName?: string;
    showRing?: boolean;
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
    ({ className, containerClassName, showRing = true, ...props }, ref) => {
        const [isFocused, setIsFocused] = React.useState(false);

        return (
            <div className={cn("relative", containerClassName)}>
                <textarea
                    className={cn(
                        "flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
                        "transition-all duration-200 ease-in-out",
                        "placeholder:text-muted-foreground",
                        "disabled:cursor-not-allowed disabled:opacity-50",
                        showRing
                            ? "focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
                            : "",
                        className
                    )}
                    ref={ref}
                    onFocus={() => setIsFocused(true)}
                    onBlur={() => setIsFocused(false)}
                    {...props}
                />

                {showRing && isFocused && (
                    <motion.span
                        className="absolute inset-0 rounded-md pointer-events-none ring-2 ring-offset-0 ring-violet-500/30"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.2 }}
                    />
                )}
            </div>
        );
    }
);
Textarea.displayName = "Textarea";

interface ChatResult {
    answer: string;
    sources: string[];
}

export function AnimatedAIChat() {
    const [value, setValue] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState<string>("");
    const [result, setResult] = useState<ChatResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    const { textareaRef, adjustHeight } = useAutoResizeTextarea({
        minHeight: 60,
        maxHeight: 200,
    });
    const [inputFocused, setInputFocused] = useState(false);

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            setMousePosition({ x: e.clientX, y: e.clientY });
        };
        window.addEventListener("mousemove", handleMouseMove);
        return () => window.removeEventListener("mousemove", handleMouseMove);
    }, []);

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (value.trim() && !isLoading) {
                handleSendMessage();
            }
        }
    };

    const handleSendMessage = async () => {
        const query = value.trim();
        if (!query || isLoading) return;

        setIsLoading(true);
        setError(null);
        setResult(null);
        setStatus("Sending…");

        try {
            const res = await fetch(`${API_URL}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query }),
            });

            if (!res.ok || !res.body) {
                throw new Error(`Request failed (${res.status})`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            // Parse the text/event-stream: events are separated by a blank line,
            // each payload line is prefixed with "data: ".
            while (true) {
                const { done, value: chunk } = await reader.read();
                if (done) break;
                buffer += decoder.decode(chunk, { stream: true });

                const events = buffer.split("\n\n");
                buffer = events.pop() ?? "";

                for (const evt of events) {
                    const line = evt
                        .split("\n")
                        .find((l) => l.startsWith("data: "));
                    if (!line) continue;
                    const data = JSON.parse(line.slice(6));

                    if (data.type === "status") {
                        setStatus(data.label);
                    } else if (data.type === "done") {
                        setResult({
                            answer: data.answer,
                            sources: data.sources ?? [],
                        });
                        setStatus("");
                    } else if (data.type === "error") {
                        setError(data.message);
                        setStatus("");
                    }
                }
            }

            setValue("");
            adjustHeight(true);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        } finally {
            setIsLoading(false);
            setStatus("");
        }
    };

    return (
        <div className="min-h-screen flex flex-col w-full items-center justify-center bg-transparent text-white p-6 relative overflow-hidden">
            <div className="absolute inset-0 w-full h-full overflow-hidden">
                <div className="absolute top-0 left-1/4 w-96 h-96 bg-violet-500/10 rounded-full mix-blend-normal filter blur-[128px] animate-pulse" />
                <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-indigo-500/10 rounded-full mix-blend-normal filter blur-[128px] animate-pulse delay-700" />
                <div className="absolute top-1/4 right-1/3 w-64 h-64 bg-fuchsia-500/10 rounded-full mix-blend-normal filter blur-[96px] animate-pulse delay-1000" />
            </div>
            <div className="w-full max-w-2xl mx-auto relative">
                <motion.div
                    className="relative z-10 space-y-8"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, ease: "easeOut" }}
                >
                    <div className="text-center space-y-3">
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2, duration: 0.5 }}
                            className="inline-block"
                        >
                            <h1 className="text-3xl font-medium tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white/90 to-white/40 pb-1">
                                Ask FinanceBench RAG
                            </h1>
                            <motion.div
                                className="h-px bg-gradient-to-r from-transparent via-white/20 to-transparent"
                                initial={{ width: 0, opacity: 0 }}
                                animate={{ width: "100%", opacity: 1 }}
                                transition={{ delay: 0.5, duration: 0.8 }}
                            />
                        </motion.div>
                        <motion.p
                            className="text-sm text-white/40"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.3 }}
                        >
                            Ask a question about a company filing — text only
                        </motion.p>
                    </div>

                    <motion.div
                        className="relative backdrop-blur-2xl bg-white/[0.02] rounded-2xl border border-white/[0.05] shadow-2xl"
                        initial={{ scale: 0.98 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.1 }}
                    >
                        <div className="p-4">
                            <Textarea
                                ref={textareaRef}
                                value={value}
                                onChange={(e) => {
                                    setValue(e.target.value);
                                    adjustHeight();
                                }}
                                onKeyDown={handleKeyDown}
                                onFocus={() => setInputFocused(true)}
                                onBlur={() => setInputFocused(false)}
                                placeholder="e.g. What was 3M's FY2018 capital expenditure?"
                                containerClassName="w-full"
                                className={cn(
                                    "w-full px-4 py-3",
                                    "resize-none",
                                    "bg-transparent",
                                    "border-none",
                                    "text-white/90 text-sm",
                                    "focus:outline-none",
                                    "placeholder:text-white/20",
                                    "min-h-[60px]"
                                )}
                                style={{ overflow: "hidden" }}
                                showRing={false}
                                disabled={isLoading}
                            />
                        </div>

                        <div className="p-4 border-t border-white/[0.05] flex items-center justify-between gap-4">
                            <div className="flex items-center gap-2 text-xs text-white/40">
                                <ArrowUpIcon className="w-3 h-3" />
                                <span>Enter to send · Shift+Enter for newline</span>
                            </div>

                            <motion.button
                                type="button"
                                onClick={handleSendMessage}
                                whileHover={{ scale: 1.01 }}
                                whileTap={{ scale: 0.98 }}
                                disabled={isLoading || !value.trim()}
                                className={cn(
                                    "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                                    "flex items-center gap-2",
                                    value.trim() && !isLoading
                                        ? "bg-white text-[#0A0A0B] shadow-lg shadow-white/10"
                                        : "bg-white/[0.05] text-white/40"
                                )}
                            >
                                {isLoading ? (
                                    <LoaderIcon className="w-4 h-4 animate-[spin_2s_linear_infinite]" />
                                ) : (
                                    <SendIcon className="w-4 h-4" />
                                )}
                                <span>Ask</span>
                            </motion.button>
                        </div>
                    </motion.div>

                    {/* Example prompt chips */}
                    {!result && !isLoading && (
                        <div className="flex flex-wrap items-center justify-center gap-2">
                            {EXAMPLE_PROMPTS.map((p, index) => (
                                <motion.button
                                    key={p.label}
                                    onClick={() => {
                                        setValue(p.query);
                                        setTimeout(() => adjustHeight(), 0);
                                    }}
                                    className="flex items-center gap-2 px-3 py-2 bg-white/[0.02] hover:bg-white/[0.05] rounded-lg text-sm text-white/60 hover:text-white/90 transition-all relative group"
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: index * 0.1 }}
                                >
                                    <Sparkles className="w-4 h-4" />
                                    <span>{p.label}</span>
                                </motion.button>
                            ))}
                        </div>
                    )}

                    {/* Live status while the CRAG pipeline runs */}
                    <AnimatePresence>
                        {isLoading && status && (
                            <motion.div
                                className="flex items-center justify-center gap-3 text-sm text-white/70"
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: 8 }}
                            >
                                <LoaderIcon className="w-4 h-4 animate-[spin_2s_linear_infinite]" />
                                <span>{status}</span>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Error */}
                    <AnimatePresence>
                        {error && (
                            <motion.div
                                className="rounded-xl border border-red-500/20 bg-red-500/[0.05] p-4 text-sm text-red-300"
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                            >
                                {error}
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Answer + sources */}
                    <AnimatePresence>
                        {result && (
                            <motion.div
                                className="space-y-4 rounded-2xl border border-white/[0.05] bg-white/[0.02] p-5 backdrop-blur-2xl"
                                initial={{ opacity: 0, y: 12 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                            >
                                <div className="text-sm leading-relaxed text-white/90 whitespace-pre-wrap">
                                    {result.answer}
                                </div>
                                {result.sources.length > 0 && (
                                    <div className="border-t border-white/[0.05] pt-3">
                                        <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-white/40">
                                            <FileText className="w-3 h-3" />
                                            Sources
                                        </div>
                                        <div className="flex flex-wrap gap-2">
                                            {result.sources.map((src) => (
                                                <span
                                                    key={src}
                                                    className="rounded-md bg-white/[0.04] px-2 py-1 text-xs text-white/60"
                                                >
                                                    {src}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </motion.div>
            </div>

            {inputFocused && (
                <motion.div
                    className="fixed w-[50rem] h-[50rem] rounded-full pointer-events-none z-0 opacity-[0.02] bg-gradient-to-r from-violet-500 via-fuchsia-500 to-indigo-500 blur-[96px]"
                    animate={{
                        x: mousePosition.x - 400,
                        y: mousePosition.y - 400,
                    }}
                    transition={{
                        type: "spring",
                        damping: 25,
                        stiffness: 150,
                        mass: 0.5,
                    }}
                />
            )}
        </div>
    );
}
