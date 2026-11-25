# Desafio de Navegação Autônoma — Atividade Ponderada

Solução em **ROS 2** para o desafio de navegação autônoma em labirinto utilizando o simulador [Culling Games (`rmnicola/culling_games`)](https://github.com/rmnicola/culling_games), cobrindo:

- **Parte 1 — Navegação com mapa**: obtenção do grid via serviço `/get_map` e planejamento da rota ótima até o alvo utilizando busca em grafos (**A\*** com heurística de Manhattan, com opção de **BFS**).
- **Parte 2 — Mapeamento do labirinto e comprovação**: exploração autônoma do labirinto utilizando exclusivamente sensores locais (`/culling_games/robot_sensors`) e comandos de movimento (`/move_command`), reconstrução do mapa em matriz de ocupação, reinício do mesmo labirinto (`/reset`) e execução do **mesmo algoritmo da Parte 1** sobre o mapa construído, comprovando equivalência da rota ótima.

---

## Estrutura do Repositório

```text
.
├── generate_maze.py                   # Gerador de labirintos do simulador
├── lobotomy_kaisen/                   # Imagens de vitória do simulador
└── src/
    ├── cg/                            # Simulador Pygame (culling_games)
    ├── cg_interfaces/                 # Mensagens e serviços ROS 2 (GetMap, MoveCmd, Reset, RobotSensors)
    ├── cg_teleop/                     # Teleoperação por teclado do simulador
    └── cg_navigator/                  # Pacote ROS 2 com a solução das Partes 1 e 2
        ├── cg_navigator/
        │   ├── algorithms.py          # A*, BFS, busca de fronteira, reconstrução e auditoria de mapa
        │   ├── part1_navigator.py     # Nó ROS 2 da Parte 1 (navegação com mapa)
        │   └── part2_mapper.py        # Nó ROS 2 da Parte 2 (mapeamento + comprovação)
        └── test/
            └── test_algorithms.py     # Testes unitários e simulação offline nos 11 mapas do simulador
```

---

## Pré-requisitos e Build

Requisitos: **ROS 2** (testado em ROS 2 Jazzy), `colcon`, `pygame` e `pytest`.

Caso utilize um ambiente RoboStack via `micromamba`/`conda`, ative-o antes dos comandos:

```bash
micromamba activate ros_jazzy
```

Na raiz do repositório, compile o workspace e carregue o ambiente:

```bash
colcon build
source install/setup.bash
```

---

## Como Executar

> Lembre-se de executar `source install/setup.bash` em cada novo terminal aberto.

### 1. Iniciar o simulador

Em um terminal, inicie o labirinto:

```bash
# Labirinto aleatório
ros2 run cg maze

# Ou um mapa específico (ex.: 1.csv até 10.csv, test.csv)
ros2 run cg maze -- --map 1.csv

# Ou gerar um labirinto aleatório novo
ros2 run cg maze -- --generate
```

### 2. Executar a Parte 1 (Navegação com Mapa)

Em outro terminal:

```bash
ros2 run cg_navigator part1
```

Opções disponíveis:
- `--algorithm {astar,bfs}`: seleciona o algoritmo de busca (padrão: `astar`).
- `--delay <segundos>`: intervalo entre movimentos para facilitar a visualização (padrão: `0.03`; use `0.0` para execução rápida).
- `--no-ascii`: oculta a visualização ASCII do mapa no terminal.

Exemplo com BFS:

```bash
ros2 run cg_navigator part1 --algorithm bfs --delay 0.02
```

### 3. Executar a Parte 2 (Mapeamento + Comprovação)

Com o simulador aberto (ou após reiniciá-lo com `ros2 service call /reset cg_interfaces/srv/Reset "{is_random: false, map_name: ''}"`):

```bash
ros2 run cg_navigator part2
```

Opções disponíveis:
- `--algorithm {astar,bfs}`: algoritmo da Parte 1 aplicado ao mapa construído (padrão: `astar`).
- `--explore-delay <segundos>`: atraso entre movimentos na fase de exploração (padrão: `0.0`).
- `--replay-delay <segundos>`: atraso entre movimentos na reprodução da rota ótima (padrão: `0.03`).
- `--save-map <caminho.csv>`: exporta o mapa reconstruído em formato CSV compatível com o simulador.
- `--no-verify`: desativa a auditoria pós-planejamento contra `/get_map` (útil para rodar sem nenhuma chamada a `/get_map`, nem mesmo de auditoria).
- `--no-ascii`: oculta a visualização ASCII do mapa no terminal.

Exemplo salvando o mapa reconstruído:

```bash
ros2 run cg_navigator part2 --save-map mapped_maze.csv
```

### 4. Executar os Testes Automatizados

A suíte de testes valida a otimalidade do A\* contra o BFS e executa o fluxo completo de mapeamento, reconstrução e comprovação da Parte 2 em todos os 11 mapas do simulador (`1.csv`–`10.csv` e `test.csv`), além de casos de borda.

Após o build (`colcon build`), carregue o workspace antes de rodar os testes:

```bash
source install/setup.bash
pytest -v src/cg_navigator/test/test_algorithms.py
```

Como alternativa sem compilar o workspace, defina `PYTHONPATH`:

```bash
PYTHONPATH=src/cg_navigator pytest -v src/cg_navigator/test/test_algorithms.py
```

---

## Como a Solução Funciona

### Parte 1 — Busca Ótima com A\* (e BFS)

1. O nó `cg_part1_navigator` requisita o grid completo ao serviço `/get_map` (`cg_interfaces/srv/GetMap`) e reconstrói a matriz 2D, identificando a célula inicial do robô (`'r'`) e a célula do alvo (`'t'`).
2. O labirinto é modelado como um **grafo não-ponderado 4-conectado** (movimentos `up`, `down`, `left`, `right`), onde células `'b'` (parede) são obstáculos intransponíveis e `'f'`, `'r'`, `'t'` são vértices transitáveis.
3. O algoritmo **A\*** utiliza a **distância de Manhattan** como heurística:
   $$h((r_1, c_1), (r_2, c_2)) = |r_1 - r_2| + |c_1 - c_2|$$
   Por ser **admissível** e **consistente** em grids 4-conectados de custo unitário, o A\* garante encontrar um **caminho mínimo global** expandindo menos nós que a busca em largura (BFS).
4. A sequência de coordenadas é convertida em direções e executada passo a passo via serviço `/move_command`.

### Parte 2 — Mapeamento por Fronteiras e Comprovação

1. **Exploração sem acesso ao mapa global**:
   - O nó `cg_part2_mapper` assina o tópico `/culling_games/robot_sensors` com perfil `qos_profile_sensor_data` (compatível com o publicador do simulador) e utiliza apenas `/move_command` para navegar.
   - O robô mantém sua pose em **coordenadas relativas** a partir da origem `(0, 0)` e registra, a cada célula inédita visitada, as **8 células adjacentes** reportadas pelo sensor (`up`, `down`, `left`, `right` e as 4 diagonais).
   - A estratégia de exploração é **Nearest-Frontier Exploration**: uma BFS sobre o mapa parcial conhecido encontra o caminho mais curto até a célula livre (`'f'`) ainda não visitada mais próxima.
   - **Preservação do estado do jogo durante a exploração**: quando o alvo `'t'` é detectado pelos sensores, sua posição é registrada no mapa, mas o robô **não entra na célula `'t'`** durante o mapeamento (evitando acionar `win = True` no simulador antes de concluir o mapeamento). Como qualquer rota ótima de `'r'` até `'t'` é um caminho simples que só entra em `'t'` no último passo, explorar todas as células alcançáveis sem atravessar `'t'` é matematicamente suficiente para conter todas as rotas ótimas até o alvo.
2. **Reconstrução e planejamento com o algoritmo da Parte 1**:
   - Quando não restam fronteiras alcançáveis, o mapa esparso relativo é normalizado para uma matriz 2D (`build_grid_from_map`).
   - O **mesmo algoritmo de busca da Parte 1** (`solve_grid`) é executado sobre o mapa reconstruído, gerando a rota ótima congelada antes de qualquer auditoria.
3. **Reinício e comprovação**:
   - O serviço `/reset` (`is_random=False, map_name=""`) reinicia o **mesmo labirinto** colocando o robô novamente na posição inicial.
   - A rota planejada exclusivamente sobre o mapa construído é executada passo a passo via `/move_command` até atingir o alvo `'t'`.
   - Adicionalmente (quando `--no-verify` não é passado), após a rota já estar planejada, o nó audita o mapa construído e a rota contra o gabarito da Parte 1, verificando:
     - **100% de acurácia** nas células mapeadas;
     - **Mesmo comprimento ótimo** entre a rota da Parte 1 e a rota obtida a partir do mapa mapeado;
     - **Validade completa** da rota no mapa real até o alvo.
