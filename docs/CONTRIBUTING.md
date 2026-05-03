# 📑 Padrões de Versionamento e Fluxo de Trabalho

Este projeto utiliza **Trunk Based Development (TBD)** para garantir integração contínua e velocidade de entrega.

## 1. Fluxo de Trabalho (Trunk Based Development)
- **Main Branch:** A `main` é a única branch de longa duração e deve estar sempre em estado "deployable".
- **Short-lived Branches:** Crie branches temporárias para features ou correções. Elas devem ser curtas (vida útil de poucas horas ou no máximo 1-2 dias).
- **Integração:** Faça o merge para a `main` o mais rápido possível após passar nos testes (`pytest`), verificação de tipos (`ty`) e linting (`ruff`).
- **No Long-running Features:** Use *Feature Flags* se uma funcionalidade precisar de mais tempo para ser concluída, em vez de manter uma branch aberta por muito tempo.

## 2. Padrão de Commits (Conventional Commits)
As mensagens de commit devem seguir a especificação: `<tipo>(escopo opcional): <descrição curta em minúsculas>`.

**Principais tipos:**
- `feat`: Uma nova funcionalidade.
- `fix`: Correção de um bug.
- `docs`: Alterações apenas na documentação.
- `style`: Alterações que não afetam o significado do código (espaços, formatação, falta de ponto e vírgula, etc).
- `refactor`: Alteração de código que não corrige um bug nem adiciona uma funcionalidade.
- `perf`: Alteração de código que melhora o desempenho.
- `test`: Adição de testes ausentes ou correção de testes existentes.
- `chore`: Atualizações de tarefas de build, configurações de pacotes (Poetry), etc.

*Exemplo:* `feat(auth): add jwt middleware for latency tracking`

## 3. Versionamento Semântico (SemVer 2.0.0)
As tags de versão devem seguir o formato `vMAJOR.MINOR.PATCH` (ex: `v1.2.3`):

- **MAJOR:** Versão incrementada quando há mudanças incompatíveis na API (Breaking Changes).
- **MINOR:** Versão incrementada quando se adiciona funcionalidade de maneira compatível com versões anteriores.
- **PATCH:** Versão incrementada quando se faz correções de bugs compatíveis com versões anteriores.

### Ciclo de Release e Tags:
1. Após o merge na `main`, quando um conjunto de funcionalidades estiver pronto para entrega:
2. Crie uma tag anotada no Git: `git tag -a v1.0.0 -m "Release description"`.
3. Garanta que a tag seja enviada para o repositório remoto: `git push origin --tags`.

---

## 🔄 Resumo do Fluxo para a IA
Sempre que for sugerir comandos de Git ou explicar o que foi feito:
1. **Nomeie a branch** de forma curta: `feat/nome-da-feature`.
2. **Escreva o commit** usando o padrão `feat: ...` ou `fix: ...`.
3. **Sugira o merge** imediato após a validação do código.
4. **Indique se um novo incremento de tag** (Major, Minor ou Patch) é necessário com base nas alterações realizadas.
