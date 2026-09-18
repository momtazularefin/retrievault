'use client';

import React, { useState, useEffect, useRef, useSyncExternalStore } from 'react';
import ReactMarkdown from 'react-markdown';

type Citation = {
  label: string;
  file_path: string;
  start_line: number;
  end_line: number;
  symbol_name: string;
  symbol_type: string;
  github_url: string;
  snippet: string;
};

type Grounding = {
  status: 'grounded' | 'refused' | 'invalid_citations' | 'uncited' | 'truncated';
  invalid_labels: string[];
  retries: number;
};

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  refused?: boolean;
  grounding?: Grounding;
  metadata?: {
    latency_ms: Record<string, number>;
    tokens: Record<string, number>;
    est_cost_usd: number;
  };
};

type HealthInfo = {
  status: string;
  qdrant: boolean;
  index_complete: boolean;
  model: string;
  corpus: {
    repo: string;
    commit_tag: string;
    chunk_count: number | null;
    chunker_version: string | null;
  };
  build_hash: string;
  startup_time: number;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const ACTIVE_CHAT_KEY = 'retrievault_active_chat';
const RECENT_QUERIES_KEY = 'retrievault_recent_queries';

// Hydration-safe "are we past the server render": the server snapshot is false, so the first
// client render matches the server's HTML, and React re-renders with stored state right after.
const emptySubscribe = () => () => {};
function useIsClient() {
  return useSyncExternalStore(emptySubscribe, () => true, () => false);
}

function readStoredJson<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') return fallback;
  try {
    const saved = localStorage.getItem(key);
    return saved ? (JSON.parse(saved) as T) : fallback;
  } catch (error) {
    console.error(`Failed to read ${key}`, error);
    return fallback;
  }
}

const SUGGESTIONS = [
  'How does APIRouter.include_router combine prefixes?',
  'What does _DefaultLifespan do on startup and shutdown?',
  'How to configure a Celery broker in FastAPI core?',
];

// Answers that are not plainly grounded say so, rather than looking like any other answer.
const GROUNDING_NOTICES: Record<Grounding['status'], string | null> = {
  grounded: null,
  refused: null,
  invalid_citations: 'Some citations in this answer did not resolve to a retrieved source and were dropped.',
  uncited: 'This answer cites no source, so nothing in it is backed by the retrieved code.',
  truncated: 'This answer hit the output limit and stops mid-thought.',
};

export default function ChatInterface() {
  const isClient = useIsClient();
  const [messages, setMessages] = useState<Message[]>(() =>
    readStoredJson<Message[]>(ACTIVE_CHAT_KEY, [])
  );
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [recentQueries, setRecentQueries] = useState<string[]>(() =>
    readStoredJson<string[]>(RECENT_QUERIES_KEY, [])
  );
  // Stored state is withheld from the server render and the hydration render.
  const visibleMessages = isClient ? messages : [];
  const visibleRecentQueries = isClient ? recentQueries : [];

  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [relativeTime, setRelativeTime] = useState('Checking connection...');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const nextMessageId = useRef(0);

  useEffect(() => {
    if (!isClient) return;
    try {
      if (messages.length > 0) {
        localStorage.setItem(ACTIVE_CHAT_KEY, JSON.stringify(messages));
      } else {
        localStorage.removeItem(ACTIVE_CHAT_KEY);
      }
    } catch (error) {
      console.error('Failed to save chat', error);
    }
  }, [messages, isClient]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        if (!res.ok) throw new Error('Unhealthy');
        const data: HealthInfo = await res.json();
        setHealth(data);
        setIsConnected(true);
      } catch {
        setIsConnected(false);
        setHealth(null);
      }
    };

    checkHealth();
    const healthInterval = setInterval(checkHealth, 10000);
    return () => clearInterval(healthInterval);
  }, []);

  useEffect(() => {
    if (!health) return;

    const updateTime = () => {
      const elapsed = Math.max(0, Math.floor(Date.now() / 1000) - health.startup_time);
      if (elapsed < 5) setRelativeTime('Updated just now');
      else if (elapsed < 60) setRelativeTime(`Updated ${elapsed}s ago`);
      else if (elapsed < 3600) setRelativeTime(`Updated ${Math.floor(elapsed / 60)}m ago`);
      else setRelativeTime(`Updated ${Math.floor(elapsed / 3600)}h ago`);
    };

    updateTime();
    const timeInterval = setInterval(updateTime, 1000);
    return () => clearInterval(timeInterval);
  }, [health]);

  const submitQuery = async (queryText: string) => {
    const cleanQuery = queryText.trim();
    if (!cleanQuery || isLoading) return;

    setRecentQueries(prev => {
      const updated = [cleanQuery, ...prev.filter(q => q !== cleanQuery)].slice(0, 10);
      try {
        localStorage.setItem(RECENT_QUERIES_KEY, JSON.stringify(updated));
      } catch (error) {
        console.error('Failed to save recent queries', error);
      }
      return updated;
    });

    const createMessageId = () => {
      nextMessageId.current += 1;
      const uuid = globalThis.crypto?.randomUUID?.();
      return `message-${uuid ?? `${Date.now()}-${nextMessageId.current}`}`;
    };

    const userMsg: Message = { id: createMessageId(), role: 'user', content: cleanQuery };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg.content }),
      });

      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();

      setMessages(prev => [
        ...prev,
        {
          id: createMessageId(),
          role: 'assistant',
          content: data.answer,
          citations: data.citations,
          refused: data.refused,
          grounding: data.grounding,
          metadata: data.metadata,
        },
      ]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [
        ...prev,
        {
          id: createMessageId(),
          role: 'assistant',
          content: 'Sorry, I encountered an error communicating with the backend.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submitQuery(input);
  };

  const startNewChat = () => {
    setMessages([]);
    setInput('');
  };

  const displayedRelativeTime = health
    ? relativeTime
    : isConnected === false ? 'Connection failed' : 'Checking connection...';

  return (
    <div className="flex w-full h-full overflow-hidden bg-[#090d16] text-slate-100 font-sans">

      {/* Sidebar Panel */}
      <aside className="hidden md:flex flex-col w-64 bg-[#0c1221] border-r border-white/5 p-4 justify-between h-full select-none">
        <div className="flex flex-col gap-6">
          <button
            onClick={startNewChat}
            className="flex items-center gap-2 px-2 py-1 text-left cursor-pointer group focus:outline-none"
          >
            <svg className="w-6 h-6 text-blue-400 fill-current animate-pulse group-hover:scale-110 transition-transform duration-300" viewBox="0 0 24 24">
              <path d="M12 2L9 9 2 12l7 3 3 7 3-7 7-3-7-3-3-7z" />
            </svg>
            <span className="font-display font-bold text-xl bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent group-hover:opacity-90 transition-opacity">
              RetrieVault
            </span>
          </button>

          <button
            onClick={startNewChat}
            className="flex items-center justify-center gap-2 w-full py-2.5 px-4 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 text-blue-400 font-semibold rounded-full transition-all duration-300 hover:scale-[1.01]"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            New chat
          </button>

          <div className="flex flex-col gap-2">
            <span className="text-[10px] font-bold tracking-wider text-slate-500 uppercase px-2">Recent Queries</span>
            <div className="flex flex-col gap-0.5 max-h-[300px] overflow-y-auto pr-1">
              {visibleRecentQueries.map(q => (
                <button
                  key={q}
                  onClick={() => submitQuery(q)}
                  className="text-left text-xs text-slate-400 hover:text-slate-200 hover:bg-white/5 px-2.5 py-2 rounded-lg truncate transition-all duration-200"
                  title={q}
                >
                  💬 {q}
                </button>
              ))}
              {visibleRecentQueries.length === 0 && (
                <span className="text-xs text-slate-600 px-2.5 py-2 italic">No history yet</span>
              )}
            </div>
          </div>
        </div>

        {/* Footer Info: reported by the backend, never hard-coded here. */}
        <div className="flex flex-col gap-2 border-t border-white/5 pt-4 px-2 text-[10px] text-slate-500">
          {health ? (
            <div className="flex flex-col gap-1.5">
              <span>🧠 {health.model}</span>
              <span>
                📚 {health.corpus.repo} @ {health.corpus.commit_tag}
                {health.corpus.chunk_count !== null && ` · ${health.corpus.chunk_count} chunks`}
              </span>
              <span>🛠️ Build: <code className="bg-white/5 px-1 rounded text-blue-400/90 font-mono font-semibold">{health.build_hash}</code></span>
              {!health.index_complete && <span className="text-amber-400">⚠️ Index incomplete</span>}
            </div>
          ) : (
            <span>Backend unavailable</span>
          )}
        </div>
      </aside>

      {/* Main Chat Panel */}
      <section className="flex-1 flex flex-col h-full overflow-hidden relative bg-[radial-gradient(circle_at_50%_40%,rgba(16,24,48,0.7),transparent_50%)]">

        <header className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-[#090d16]/30 backdrop-blur-sm z-10 select-none">
          <button
            onClick={startNewChat}
            className="flex items-center gap-2 md:hidden text-left focus:outline-none group"
          >
            <svg className="w-5 h-5 text-blue-400 fill-current animate-pulse group-hover:scale-105 transition-transform duration-300" viewBox="0 0 24 24">
              <path d="M12 2L9 9 2 12l7 3 3 7 3-7 7-3-7-3-3-7z" />
            </svg>
            <span className="font-display font-bold text-lg bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent group-hover:opacity-90 transition-opacity">
              RetrieVault
            </span>
          </button>
          <div className="hidden md:flex items-center gap-2 text-xs font-semibold tracking-wider text-slate-500 uppercase">
            <span>FastAPI Codebase RAG</span>
          </div>
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-rose-500 animate-pulse'}`} />
            <span className="text-xs text-slate-400 font-medium">{isConnected ? 'Server Online' : 'Offline'}</span>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto w-full">
          <div className="max-w-4xl mx-auto px-6 py-8 md:py-12 flex flex-col gap-8 w-full min-h-full">
            {visibleMessages.length === 0 ? (
              <div className="flex-grow flex flex-col justify-center items-center text-center animate-[fadeIn_0.5s_ease-out_forwards] gap-10 py-12 select-none">
                <div className="flex flex-col gap-3">
                  <h2 className="font-display text-3xl md:text-5xl font-bold bg-gradient-to-r from-blue-300 via-blue-100 to-indigo-300 bg-clip-text text-transparent leading-tight">
                    Ask the FastAPI source
                  </h2>
                  <p className="text-slate-400 text-sm md:text-base max-w-md mx-auto">
                    Every answer is built from indexed source chunks and cites the exact file and
                    lines. Questions the code cannot answer are refused, not guessed.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 w-full max-w-3xl mt-4">
                  {SUGGESTIONS.map(s => (
                    <button
                      key={s}
                      onClick={() => submitQuery(s)}
                      className="text-left text-sm bg-[#101625]/60 hover:bg-[#151e33] border border-white/5 hover:border-blue-500/20 p-4 rounded-xl shadow-md transition-all duration-300 hover:scale-[1.01] hover:-translate-y-0.5 group flex flex-col justify-between h-28"
                    >
                      <span className="text-slate-300 font-medium leading-relaxed">{s}</span>
                      <span className="text-[11px] font-bold tracking-wider text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        Run Query →
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex flex-col gap-8 pb-12">
                {visibleMessages.map(m => (
                  <div
                    key={m.id}
                    className={`flex gap-4 w-full animate-[fadeIn_0.3s_ease-out] ${
                      m.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    {m.role === 'assistant' && (
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-md select-none">
                        <svg className="w-4 h-4 text-white fill-current" viewBox="0 0 24 24">
                          <path d="M12 2L9 9 2 12l7 3 3 7 3-7 7-3-7-3-3-7z" />
                        </svg>
                      </div>
                    )}

                    <div className={`flex flex-col max-w-[85%] ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
                      <div className={`px-5 py-3.5 rounded-2xl text-[15px] leading-relaxed ${
                        m.role === 'user'
                          ? 'bg-blue-600 text-white rounded-tr-sm shadow-md'
                          : 'text-slate-200 bg-transparent w-full'
                      }`}>
                        {m.role === 'user' ? (
                          m.content
                        ) : (
                          <div className="flex flex-col gap-2">
                            {m.refused && (
                              <span className="self-start text-[10px] font-bold tracking-wider uppercase text-amber-400 bg-amber-400/10 border border-amber-400/25 px-2 py-0.5 rounded">
                                Not in the retrieved source
                              </span>
                            )}
                            <MarkdownWithCitations content={m.content} citations={m.citations} />
                            {m.grounding && GROUNDING_NOTICES[m.grounding.status] && (
                              <span className="text-xs text-amber-400/90">
                                ⚠️ {GROUNDING_NOTICES[m.grounding.status]}
                                {m.grounding.invalid_labels.length > 0 &&
                                  ` (${m.grounding.invalid_labels.join(', ')})`}
                              </span>
                            )}
                          </div>
                        )}
                      </div>

                      {m.metadata && (
                        <div className="mt-1 px-5 text-[10px] text-slate-500 flex gap-4 select-none">
                          <span>⏱️ {m.metadata.latency_ms.total.toFixed(0)}ms</span>
                          <span>🪙 ${m.metadata.est_cost_usd.toFixed(4)}</span>
                          <span>Tokens: {m.metadata.tokens.input} in / {m.metadata.tokens.output} out</span>
                          {m.grounding && m.grounding.retries > 0 && <span>↻ 1 corrective retry</span>}
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="flex gap-4 w-full justify-start animate-[fadeIn_0.3s_ease-out]">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-md select-none">
                      <svg className="w-4 h-4 text-white fill-current animate-spin" viewBox="0 0 24 24">
                        <path d="M12 2L9 9 2 12l7 3 3 7 3-7 7-3-7-3-3-7z" />
                      </svg>
                    </div>
                    <div className="px-5 py-3.5 text-sm text-slate-400 italic">
                      Reading codebase...
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        <div className="w-full bg-gradient-to-t from-[#090d16] via-[#090d16]/95 to-transparent pt-4 pb-4 px-4 z-10">
          <div className="max-w-4xl mx-auto w-full flex flex-col gap-2.5">
            <form onSubmit={handleSubmit} className="flex items-center gap-3 bg-[#111726]/80 hover:bg-[#151d30] border border-white/5 focus-within:border-blue-500/30 rounded-full px-5 py-3.5 shadow-lg backdrop-blur-md transition-all duration-300">
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="Ask about dependency scopes, routers, or request handling..."
                disabled={isLoading}
                maxLength={2000}
                className="flex-grow bg-transparent text-white placeholder-slate-500 text-base outline-none font-sans px-2"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 hover:bg-blue-600 disabled:opacity-30 disabled:hover:bg-blue-500 text-white flex items-center justify-center transition-all duration-300 hover:scale-[1.03] disabled:scale-[1.0]"
              >
                <svg className="w-4 h-4 fill-current transform rotate-90" viewBox="0 0 24 24">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                </svg>
              </button>
            </form>

            <div className="flex items-center justify-between px-6 text-[10px] text-slate-500 select-none">
              <div className="flex items-center gap-2">
                <span>API Status: </span>
                <span className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.4)]' : 'bg-rose-500 animate-pulse'}`} />
                <span className="font-semibold text-slate-400">{isConnected ? 'Connected' : 'Offline'}</span>
              </div>
              <div>
                <span>{displayedRelativeTime}</span>
              </div>
            </div>
          </div>
        </div>

      </section>
    </div>
  );
}

function MarkdownWithCitations({ content, citations }: { content: string, citations?: Citation[] }) {
  let processedContent = content;
  if (citations && citations.length > 0) {
    citations.forEach(cit => {
      // cit.label is "[S3]"; link only that exact label, so [S3] never matches inside [S30].
      const number = cit.label.replace(/\D/g, '');
      const regex = new RegExp(`\\[S${number}\\]`, 'g');
      processedContent = processedContent.replace(regex, `[${cit.label}](${cit.github_url})`);
    });
  }

  return (
    <ReactMarkdown
      components={{
        p: ({children}) => <p className="mt-0 text-[15px] md:text-[16px] leading-relaxed font-sans font-medium text-slate-100 antialiased tracking-wide">{children}</p>,
        h1: ({children}) => <h1 className="text-2xl font-bold font-display text-white mt-6 mb-2 tracking-wide">{children}</h1>,
        h2: ({children}) => <h2 className="text-xl font-bold font-display text-white mt-5 mb-2 tracking-wide">{children}</h2>,
        h3: ({children}) => <h3 className="text-lg font-bold font-display text-white mt-4 mb-2 tracking-wide">{children}</h3>,
        ul: ({children}) => <ul className="list-disc pl-6 my-3 flex flex-col gap-2 font-sans text-slate-100 text-[15px] md:text-[16px] font-medium tracking-wide leading-relaxed">{children}</ul>,
        ol: ({children}) => <ol className="list-decimal pl-6 my-3 flex flex-col gap-2 font-sans text-slate-100 text-[15px] md:text-[16px] font-medium tracking-wide leading-relaxed">{children}</ol>,
        li: ({children}) => <li className="leading-relaxed">{children}</li>,
        pre: ({children}) => <pre className="bg-[#0b0f19] p-4 rounded-xl overflow-x-auto border border-white/5 shadow-inner mt-2 font-mono text-sm text-blue-200/90 leading-relaxed">{children}</pre>,
        code: ({className, children}) => {
          if (className) {
            return <code className="bg-transparent p-0 font-mono text-sm">{children}</code>;
          }
          return <code className="font-mono bg-white/5 px-1.5 py-0.5 rounded-md text-blue-300 text-sm border border-white/5">{children}</code>;
        },
        a: ({ href, children }) => {
          const isCitation = children?.toString().startsWith('[S');
          if (isCitation) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center bg-blue-400/10 text-blue-400 border border-blue-400/25 px-1.5 py-0.5 rounded text-[11px] font-semibold mx-0.5 no-underline transition-all hover:bg-blue-400/20 hover:-translate-y-0.5 align-super"
                title={href}
              >
                {children}
              </a>
            );
          }
          return <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">{children}</a>;
        }
      }}
    >
      {processedContent}
    </ReactMarkdown>
  );
}
