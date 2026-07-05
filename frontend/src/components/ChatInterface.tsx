'use client';

import React, { useState, useEffect, useRef } from 'react';
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

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  metadata?: {
    latency_ms: Record<string, number>;
    tokens: Record<string, number>;
    est_cost_usd: number;
  };
};

type HealthInfo = {
  status: string;
  qdrant: boolean;
  model: string;
  corpus: {
    repo: string;
    commit_tag: string;
    chunk_count: number | null;
  };
  build_hash: string;
  startup_time: number;
};

function readStoredJson<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') return fallback;

  const saved = localStorage.getItem(key);
  if (!saved) return fallback;

  try {
    return JSON.parse(saved) as T;
  } catch (error) {
    console.error(`Failed to parse ${key}`, error);
    return fallback;
  }
}

const SUGGESTIONS = [
  "How to use FastAPI's Cookie to declare a default value?",
  "What is the purpose of the _DefaultLifespan class?",
  "Does FastAPI support Flask blueprints?"
];

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>(() =>
    readStoredJson<Message[]>('retrievault_active_chat', [])
  );
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [recentQueries, setRecentQueries] = useState<string[]>(() =>
    readStoredJson<string[]>('retrievault_recent_queries', [])
  );
  
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [relativeTime, setRelativeTime] = useState('Checking connection...');
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const nextMessageId = useRef(0);

  // Sync messages to localStorage
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('retrievault_active_chat', JSON.stringify(messages));
    } else {
      localStorage.removeItem('retrievault_active_chat');
    }
  }, [messages]);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    const checkHealth = async () => {
      try {
        const res = await fetch(`${apiUrl}/health`);
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
    if (!health) {
      return;
    }

    const updateTime = () => {
      const nowSeconds = Math.floor(Date.now() / 1000);
      const elapsed = Math.max(0, nowSeconds - health.startup_time);

      if (elapsed < 5) {
        setRelativeTime('Updated just now');
      } else if (elapsed < 60) {
        setRelativeTime(`Updated ${elapsed}s ago`);
      } else if (elapsed < 3600) {
        setRelativeTime(`Updated ${Math.floor(elapsed / 60)}m ago`);
      } else {
        setRelativeTime(`Updated ${Math.floor(elapsed / 3600)}h ago`);
      }
    };

    updateTime();
    const timeInterval = setInterval(updateTime, 1000);
    return () => clearInterval(timeInterval);
  }, [health]);

  const submitQuery = async (queryText: string) => {
    const cleanQuery = queryText.trim();
    if (!cleanQuery || isLoading) return;

    // Track unique recent queries in state & localStorage
    setRecentQueries(prev => {
      const updated = [cleanQuery, ...prev.filter(q => q !== cleanQuery)].slice(0, 10);
      localStorage.setItem('retrievault_recent_queries', JSON.stringify(updated));
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
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg.content }),
      });

      if (!res.ok) throw new Error('API Error');
      const data = await res.json();

      const assistantMsg: Message = {
        id: createMessageId(),
        role: 'assistant',
        content: data.answer,
        citations: data.citations,
        metadata: data.metadata,
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      console.error(err);
      const errorMsg: Message = {
        id: createMessageId(),
        role: 'assistant',
        content: 'Sorry, I encountered an error communicating with the backend.'
      };
      setMessages(prev => [...prev, errorMsg]);
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
          {/* Logo - Clickable to Landing Page */}
          <button 
            onClick={startNewChat}
            className="flex items-center gap-2 px-2 py-1 text-left cursor-pointer group focus:outline-none"
          >
            <svg className="w-6 h-6 text-blue-400 fill-current animate-pulse group-hover:scale-110 transition-transform duration-300" viewBox="0 0 24 24">
              <path d="M12 2L9 9 2 12l7 3 3 7 3-7 7-3-7-3-3-7z" />
            </svg>
            <span className="font-display font-bold text-xl bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent group-hover:opacity-90 transition-opacity">
              retrievault
            </span>
          </button>

          {/* New Chat Button */}
          <button 
            onClick={startNewChat}
            className="flex items-center justify-center gap-2 w-full py-2.5 px-4 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 text-blue-400 font-semibold rounded-full transition-all duration-300 hover:scale-[1.01]"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            New chat
          </button>

          {/* Recent Queries */}
          <div className="flex flex-col gap-2">
            <span className="text-[10px] font-bold tracking-wider text-slate-500 uppercase px-2">Recent Queries</span>
            <div className="flex flex-col gap-0.5 max-h-[300px] overflow-y-auto pr-1">
              {recentQueries.map((q, idx) => (
                <button 
                  key={idx} 
                  onClick={() => submitQuery(q)}
                  className="text-left text-xs text-slate-400 hover:text-slate-200 hover:bg-white/5 px-2.5 py-2 rounded-lg truncate transition-all duration-200"
                  title={q}
                >
                  💬 {q}
                </button>
              ))}
              {recentQueries.length === 0 && (
                <span className="text-xs text-slate-600 px-2.5 py-2 italic">No history yet</span>
              )}
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <div className="flex flex-col gap-2 border-t border-white/5 pt-4 px-2 text-[10px] text-slate-500">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
            <span>LLM: Claude Sonnet 4.6</span>
          </div>
          {health && (
            <div className="flex flex-col gap-1.5">
              <span>📚 {health.corpus.repo} @ {health.corpus.commit_tag}</span>
              <span>🛠️ Build: <code className="bg-white/5 px-1 rounded text-blue-400/90 font-mono font-semibold">{health.build_hash}</code></span>
            </div>
          )}
        </div>
      </aside>

      {/* Main Chat Panel */}
      <section className="flex-1 flex flex-col h-full overflow-hidden relative bg-[radial-gradient(circle_at_50%_40%,rgba(16,24,48,0.7),transparent_50%)]">
        
        {/* Top Floating Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-[#090d16]/30 backdrop-blur-sm z-10 select-none">
          <button 
            onClick={startNewChat}
            className="flex items-center gap-2 md:hidden text-left focus:outline-none group"
          >
            <svg className="w-5 h-5 text-blue-400 fill-current animate-pulse group-hover:scale-105 transition-transform duration-300" viewBox="0 0 24 24">
              <path d="M12 2L9 9 2 12l7 3 3 7 3-7 7-3-7-3-3-7z" />
            </svg>
            <span className="font-display font-bold text-lg bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent group-hover:opacity-90 transition-opacity">
              retrievault
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

        {/* Scrollable Message Container */}
        <div className="flex-1 overflow-y-auto w-full">
          <div className="max-w-4xl mx-auto px-6 py-8 md:py-12 flex flex-col gap-8 w-full min-h-full">
            {messages.length === 0 ? (
              /* Welcome / Empty State */
              <div className="flex-grow flex flex-col justify-center items-center text-center animate-[fadeIn_0.5s_ease-out_forwards] gap-10 py-12 select-none">
                <div className="flex flex-col gap-3">
                  <h2 className="font-display text-3xl md:text-5xl font-bold bg-gradient-to-r from-blue-300 via-blue-100 to-indigo-300 bg-clip-text text-transparent leading-tight">
                    The codebase is yours, Momtazul.
                  </h2>
                  <p className="text-slate-400 text-sm md:text-base max-w-md mx-auto">
                    Ask questions about FastAPI&apos;s routes, dependencies, security layers, or internal configurations.
                  </p>
                </div>

                {/* Suggestions Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 w-full max-w-3xl mt-4">
                  {SUGGESTIONS.map((s, idx) => (
                    <button
                      key={idx}
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
              /* Messages Stream */
              <div className="flex flex-col gap-8 pb-12">
                {messages.map(m => (
                  <div 
                    key={m.id} 
                    className={`flex gap-4 w-full animate-[fadeIn_0.3s_ease-out] ${
                      m.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    {/* Left Avatar for Assistant */}
                    {m.role === 'assistant' && (
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-md select-none">
                        <svg className="w-4 h-4 text-white fill-current" viewBox="0 0 24 24">
                          <path d="M12 2L9 9 2 12l7 3 3 7 3-7 7-3-7-3-3-7z" />
                        </svg>
                      </div>
                    )}

                    {/* Message Bubble */}
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
                            <MarkdownWithCitations content={m.content} citations={m.citations} />
                          </div>
                        )}
                      </div>

                      {/* Metadata / Latency Indicator */}
                      {m.metadata && (
                        <div className="mt-1 px-5 text-[10px] text-slate-500 flex gap-4 select-none">
                          <span>⏱️ {m.metadata.latency_ms.total.toFixed(0)}ms</span>
                          <span>🪙 ${m.metadata.est_cost_usd.toFixed(4)}</span>
                          <span>Tokens: {m.metadata.tokens.input} in / {m.metadata.tokens.output} out</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {/* Loader */}
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

        {/* Bottom Centered Pill Input Bar */}
        <div className="w-full bg-gradient-to-t from-[#090d16] via-[#090d16]/95 to-transparent pt-4 pb-4 px-4 z-10">
          <div className="max-w-4xl mx-auto w-full flex flex-col gap-2.5">
            <form onSubmit={handleSubmit} className="flex items-center gap-3 bg-[#111726]/80 hover:bg-[#151d30] border border-white/5 focus-within:border-blue-500/30 rounded-full px-5 py-3.5 shadow-lg backdrop-blur-md transition-all duration-300">
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="Ask about dependency scopes, routers, or request handling..."
                disabled={isLoading}
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

            {/* Bottom Status / Relative Time indicator */}
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
      const regex = new RegExp(`\\[${cit.label.replace('[', '').replace(']', '')}\\]`, 'g');
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
