FROM debian:stable-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tinyproxy \
    && rm -rf /var/lib/apt/lists/*

COPY proxy/tinyproxy.conf /etc/tinyproxy/tinyproxy.conf
COPY proxy/entrypoint.sh /usr/local/bin/tinyproxy-entrypoint

RUN chmod +x /usr/local/bin/tinyproxy-entrypoint

EXPOSE 8888

USER tinyproxy

ENTRYPOINT ["tinyproxy-entrypoint"]
