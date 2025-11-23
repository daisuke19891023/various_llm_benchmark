FROM postgres:16

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg; \
    curl -fsSL https://packages.groonga.org/debian/groonga-archive-keyring.gpg \
        -o /usr/share/keyrings/groonga-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/groonga-archive-keyring.gpg] https://packages.groonga.org/debian/ bookworm main" \
        > /etc/apt/sources.list.d/groonga.list; \
    echo "deb [signed-by=/usr/share/keyrings/groonga-archive-keyring.gpg] https://packages.groonga.org/debian/ bookworm-pgdg main" \
        >> /etc/apt/sources.list.d/groonga.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        postgresql-16-pgvector \
        postgresql-16-pgroonga; \
    rm -rf /var/lib/apt/lists/*

COPY docker/initdb.d /docker-entrypoint-initdb.d
