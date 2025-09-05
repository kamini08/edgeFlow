/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // Proxy API requests to backend service when running behind Docker Compose
    return [
      {
        source: '/api/:path*',
        destination: 'http://backend:8000/api/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
