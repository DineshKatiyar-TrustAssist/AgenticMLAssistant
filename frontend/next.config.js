/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone', 
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        // This connects to the backend running locally in the same container
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
    ]
  },
}

module.exports = nextConfig
