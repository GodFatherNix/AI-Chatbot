// Temporary module stubs to satisfy TypeScript in environments where
// node_modules/@types/* packages are not available at analysis time.
// They will be overridden by real type declarations once `npm install` is run.

declare module "react" {
  const React: any;
  export default React;
}

declare module "react-dom" {
  const ReactDOM: any;
  export default ReactDOM;
}

declare module "react/jsx-runtime" {
  const jsxRuntime: any;
  export default jsxRuntime;
}

declare module "vite/client" {
  const viteClient: any;
  export default viteClient;
}

// Minimal NodeJS definitions
// eslint-disable-next-line @typescript-eslint/no-namespace
declare namespace NodeJS {
  // eslint-disable-next-line @typescript-eslint/consistent-type-definitions
  interface ProcessEnv {
    [key: string]: string | undefined;
  }
}