module.exports = {
  presets: [
    ['next/babel', { 'preset-env': {}, 'transform-runtime': {} }],
    ['@babel/preset-env', { targets: { node: 'current' } }],
    ['@babel/preset-react', { runtime: 'automatic' }],
    '@babel/preset-typescript',
  ],
};

