'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

type Citation = {
  label: string;
  file_path: string;
  start_line: number;
  end_line: number;
  symbol_name: string;
  github_url: string;
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

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input.trim() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg.content }),
      });

      if (!res.ok) throw new Error('API Error');
      const data = await res.json();

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
        citations: data.citations,
        metadata: data.metadata,
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      console.error(err);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error communicating with the backend.'
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <div className="flex-grow overflow-y-auto flex flex-col gap-6 p-8 mb-8 bg-slate-800/70 backdrop-blur-md border border-white/10 rounded-2xl shadow-[0_8px_32px_0_rgba(0,0,0,0.3)]">
        {messages.length === 0 && (
          <div className="text-center text-slate-400">
            Ask a question about the FastAPI codebase!
          </div>
        )}
        {messages.map(m => (
          <div key={m.id} className={`flex flex-col max-w-[85%] animate-[fadeIn_0.4s_ease-out_forwards] ${m.role === 'user' ? 'self-end' : 'self-start'}`}>
            <div className={`px-5 py-4 rounded-2xl leading-relaxed ${
              m.role === 'user' 
                ? 'bg-blue-500 text-white rounded-br-sm' 
                : 'bg-slate-800/70 border border-white/10 rounded-bl-sm'
            }`}>
              {m.role === 'user' ? (
                m.content
              ) : (
                <div className="flex flex-col gap-2">
                  <MarkdownWithCitations content={m.content} citations={m.citations} />
                </div>
              )}
            </div>
            {m.metadata && (
              <div className="mt-2 text-xs text-slate-400 flex gap-4">
                <span>⏱️ {m.metadata.latency_ms.total}ms</span>
                <span>🪙 ${m.metadata.est_cost_usd.toFixed(4)}</span>
                <span>Tokens: {m.metadata.tokens.input} In / {m.metadata.tokens.output} Out</span>
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="flex flex-col max-w-[85%] animate-[fadeIn_0.4s_ease-out_forwards] self-start">
            <div className="px-5 py-4 rounded-2xl leading-relaxed bg-slate-800/70 border border-white/10 rounded-bl-sm">
              Thinking...
            </div>
          </div>
        )}
      </div>
      <form className="flex gap-4 mb-4 relative" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="e.g. How does FastAPI handle dependency injection?"
          disabled={isLoading}
          className="flex-grow bg-slate-800/70 border border-white/10 rounded-full px-6 py-4 text-white font-sans outline-none focus:border-blue-500 transition-colors duration-300"
        />
        <button 
          type="submit" 
          disabled={isLoading || !input.trim()}
          className="bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-full px-6 font-semibold transition-all hover:-translate-y-px hover:shadow-[0_4px_12px_rgba(59,130,246,0.4)] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none"
        >
          Send
        </button>
      </form>
    </>
  );
}

function MarkdownWithCitations({ content, citations }: { content: string, citations?: Citation[] }) {
  // Pre-process content: replace [S1] with markdown links
  let processedContent = content;
  if (citations && citations.length > 0) {
    citations.forEach(cit => {
      // Escape brackets for regex
      const regex = new RegExp(`\\[${cit.label.replace('[', '').replace(']', '')}\\]`, 'g');
      processedContent = processedContent.replace(regex, `[${cit.label}](${cit.github_url})`);
    });
  }

  return (
    <ReactMarkdown
      components={{
        p: ({node, children}) => <p className="mt-0">{children}</p>,
        pre: ({node, children}) => <pre className="bg-black p-4 rounded-lg overflow-x-auto border border-white/10">{children}</pre>,
        code: ({node, className, children}) => {
          if (className) {
            return <code className="bg-transparent p-0">{children}</code>;
          }
          return <code className="font-mono bg-white/10 px-1.5 py-0.5 rounded-md">{children}</code>;
        },
        a: ({ node, href, children }) => {
          // If it's one of our citation links, render as a badge
          const isCitation = children?.toString().startsWith('[S');
          if (isCitation) {
            return (
              <a 
                href={href} 
                target="_blank" 
                rel="noopener noreferrer" 
                className="inline-flex items-center justify-center bg-blue-400/20 text-blue-400 border border-blue-400/30 px-1.5 py-0.5 rounded text-xs font-semibold mx-1 no-underline transition-all hover:bg-blue-400/40 hover:-translate-y-0.5 hover:shadow-[0_2px_8px_rgba(96,165,250,0.4)] align-super" 
                title={href}
              >
                {children}
              </a>
            );
          }
          return <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-500">{children}</a>;
        }
      }}
    >
      {processedContent}
    </ReactMarkdown>
  );
}
