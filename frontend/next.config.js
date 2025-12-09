/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  async rewrites() {
    // Proxy API calls to backend running in same container
    // Backend is only accessible via localhost (127.0.0.1) from within container
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
    ]
  },
}

module.exports = nextConfig
