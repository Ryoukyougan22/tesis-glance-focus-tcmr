import tseslint from 'typescript-eslint';
import globals from 'globals';
export default [
  {ignores: ['dist/**', 'node_modules/**', 'test-results*/**']},
  {files: ['src/**/*.{js,jsx,ts,tsx}'], languageOptions: {
    parser: tseslint.parser, parserOptions: {ecmaFeatures: {jsx:true}},
    globals: globals.browser,
  }, plugins: {'@typescript-eslint': tseslint.plugin}, rules: {
    'no-constant-condition':'error', 'no-unreachable':'error',
    'no-duplicate-imports':'error', '@typescript-eslint/no-explicit-any':'error',
  }},
];
