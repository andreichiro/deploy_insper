# Pra que serve essa pasta?

Aqui ficam os arquivos de configuração usados pelo Kedro e por outras ferramentas.

## Configuração local

A pasta `local` é pra config específica do usuário (ex: IDE) ou protegida (ex: chaves de segurança).

> *Nota:* nunca commite nada da pasta local no git.

## Configuração base

A pasta `base` é pra config compartilhada — coisas não-sensíveis e relacionadas ao projeto que podem ser compartilhadas entre membros do time.

AVISO: nunca coloque credenciais na pasta base.

## Saiba mais

Documentação de configuração do Kedro: [user guide](https://docs.kedro.org/en/stable/configuration/configuration_basics.html).
