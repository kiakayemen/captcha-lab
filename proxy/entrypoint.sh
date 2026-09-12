#!/bin/sh
set -eu

runtime_config=/tmp/tinyproxy.conf
cp /etc/tinyproxy/tinyproxy.conf "$runtime_config"

username=${TINYPROXY_USERNAME:-}
password=${TINYPROXY_PASSWORD:-}

if [ -n "$username" ] || [ -n "$password" ]; then
    if [ -z "$username" ] || [ -z "$password" ]; then
        echo "TINYPROXY_USERNAME and TINYPROXY_PASSWORD must be set together." >&2
        exit 1
    fi

    case "$username$password" in
        *[!A-Za-z0-9._~-]*)
            echo "Tinyproxy credentials may contain only letters, numbers, '.', '_', '~', and '-'." >&2
            exit 1
            ;;
    esac

    printf '\nBasicAuth %s %s\n' "$username" "$password" >> "$runtime_config"
fi

exec tinyproxy -d -c "$runtime_config"
