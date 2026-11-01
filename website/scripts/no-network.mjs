// Local check that the build needs no network, preloaded in Node with NODE_OPTIONS=--import:
// connections and name lookups outside loopback throw. Best effort: Bun processes (the assembly
// step) ignore it. CI runs the build in an empty network namespace instead (docs.yml).
import dns from "node:dns";
import { syncBuiltinESMExports } from "node:module";
import net from "node:net";

const loopback = new Set(["localhost", "127.0.0.1", "::1"]);
const refuse = (what) => {
  throw new Error(`no network during the build: ${what}`);
};

const connect = net.Socket.prototype.connect;
net.Socket.prototype.connect = function (...args) {
  // The same normalisation as Node's own overloads: (options), (port, host), (path), [args].
  const [options] = net._normalizeArgs(Array.isArray(args[0]) ? args[0] : args);
  if (
    typeof options.path !== "string" &&
    !loopback.has(options.host ?? "localhost")
  ) {
    refuse(`${options.host}:${options.port}`);
  }
  return connect.apply(this, args);
};

const names = [
  "lookup",
  "lookupService",
  "resolve",
  "resolve4",
  "resolve6",
  "resolveAny",
];
names.push(
  "resolveCname",
  "resolveMx",
  "resolveNs",
  "resolveSrv",
  "resolveTxt",
  "reverse",
);
for (const api of [
  dns,
  dns.promises,
  dns.Resolver.prototype,
  dns.promises.Resolver.prototype,
]) {
  for (const name of names) {
    const original = api[name];
    if (typeof original !== "function") continue;
    api[name] = function (host, ...rest) {
      if (!loopback.has(host)) refuse(`${name} ${host}`);
      return original.call(this, host, ...rest);
    };
  }
}
syncBuiltinESMExports();
