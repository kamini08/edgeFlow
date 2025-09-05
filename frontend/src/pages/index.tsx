import React from 'react';
import Link from 'next/link';

export default function Home() {
  return (
    <main style={{ padding: 24 }}>
      <h1>EdgeFlow Web</h1>
      <ul>
        <li><Link href="/compile">Compile</Link></li>
        <li><Link href="/results">Results</Link></li>
      </ul>
    </main>
  );
}

