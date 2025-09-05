import React from 'react';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b bg-white">
        <div className="container-narrow flex items-center justify-between py-4">
          <h1 className="text-xl font-semibold text-gray-900">EdgeFlow</h1>
          <nav className="space-x-4 text-sm font-medium">
            <Link className="text-gray-700 hover:text-blue-600" href="/compile">Compile</Link>
            <Link className="text-gray-700 hover:text-blue-600" href="/results">Results</Link>
          </nav>
        </div>
      </header>
      <main className="container-narrow py-10">
        <section className="card">
          <h2 className="mb-2 text-lg font-semibold text-gray-900">Welcome</h2>
          <p className="text-gray-600">
            Use the Compile page to upload and validate an EdgeFlow configuration file,
            and the Results page to view sample optimization output.
          </p>
          <div className="mt-4 flex gap-3">
            <Link className="btn" href="/compile">Go to Compile</Link>
            <Link className="btn bg-gray-700 hover:bg-gray-800" href="/results">View Results</Link>
          </div>
        </section>
      </main>
    </div>
  );
}
