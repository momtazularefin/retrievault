import ChatInterface from '@/components/ChatInterface';

export default function Home() {
  return (
    <main className="w-full max-w-4xl mx-auto p-4 flex flex-col flex-grow h-screen">
      <header className="text-center mb-8 mt-8">
        <h1 className="font-display text-4xl bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent m-0">
          retrievault
        </h1>
      </header>
      <ChatInterface />
    </main>
  );
}
