// Preloaded with NODE_OPTIONS=--import to build the site offline: every connection or name
// lookup to a host other than loopback throws, so a build step that needs the network fails
// instead of depending on it.
import dns from "node:dns";
import net from "node:net";

const loopback = new Set(["localhost", "127.0.0.1", "::1"]);
const refuse = (host) => {
  throw new Error(`no network during the build: ${host}`);
};

const connect = net.Socket.prototype.connect;
net.Socket.prototype.connect = function (...args) {
  const options =
    typeof args[0] === "object" && args[0] !== null
      ? args[0]
      : { host: args[1] };
  if (options.path === undefined && !loopback.has(options.host ?? "localhost"))
    refuse(options.host);
  return connect.apply(this, args);
};
for (const api of [dns, dns.promises]) {
  const lookup = api.lookup;
  api.lookup = function (host, ...rest) {
    if (!loopback.has(host)) refuse(host);
    return lookup.call(this, host, ...rest);
  };
}
