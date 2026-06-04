Existem apenas 6 avaliações RAG.¶
O mundo da avaliação RAG parece desnecessariamente complexo. Todo mundo está criando estruturas, métricas e gerando painéis que dão a impressão de que você precisa de um doutorado só para saber se o seu sistema está funcionando.

Mas e se eu lhe dissesse que existem apenas 6 maneiras fundamentais de avaliar um sistema RAG?

Simplesmente matemática e simetria.

A Visão¶
Os sistemas RAG possuem três componentes principais:

Uma pergunta (Q)
Contexto recuperado (C)
Uma resposta (A)
É isso. Três variáveis.

Exaustivo por natureza.

O poder de focar em Pergunta (P), Contexto (C) e Resposta (R) reside no fato de que esses três componentes, e suas relações condicionais, abrangem todos os aspectos possíveis da avaliação RAG. Não há variáveis ​​ocultas.

Se analisarmos isso sob a perspectiva das relações condicionais — a qualidade de um componente dada a condição de outro — obtemos exatamente seis relações possíveis. Nem mais, nem menos.

Pense nisso: quando algo falha no seu sistema RAG, é sempre uma dessas relações que está deixando de funcionar. Isso se relaciona diretamente com o que aprendemos sobre a abordagem sistemática para aprimorar o RAG: identificar pontos de falha específicos em vez de fazer afirmações vagas sobre "tornar a IA melhor".

As 6 métricas principais de avaliação¶
Vamos analisar cada relação (usarei a notação X|Y para significar "qualidade de X dado Y"). Em vez de tratar todas as métricas da mesma forma, vou organizá-las em três níveis práticos com base na complexidade de implementação e no impacto nos negócios — semelhante à maneira como pensamos sobre o ciclo virtuoso de melhoria para produtos RAG:

Nível 1: Métricas Fundamentais (Antes da Avaliação RAG)¶
Antes mesmo de abordarmos nossas seis métricas principais, precisamos reconhecer o fundamento de qualquer sistema de recuperação de dados:

Precisão e revocação da recuperação : Essas métricas tradicionais de recuperação de informação medem a eficiência com que seu recuperador encontra documentos relevantes no corpus. Elas são rápidas de calcular, não exigem modelos de lógica latente (LLMs) e fornecem feedback rápido para o ajuste do recuperador.

Isso está de acordo com nossa abordagem para iniciar o ciclo virtuoso com dados sintéticos: é necessário estabelecer um desempenho de recuperação de dados de referência com métricas claras e mensuráveis ​​antes de passar para técnicas de avaliação mais complexas. Essas métricas servem como indicadores preditivos que antecipam o sucesso futuro, em vez de indicadores retrospectivos que apenas informam sobre o desempenho passado.

Nível 2: Relações RAG primárias¶
1. Relevância do Contexto (C|Q)

Definição : Quão bem os trechos recuperados atendem às necessidades de informação da pergunta? Isso mede se o seu componente de recuperação está cumprindo sua função — encontrar trechos que contenham informações relevantes para responder à pergunta do usuário.

Exemplo (Bom) :

Pergunta: "Quais são os benefícios da meditação para a saúde?"

Contexto: "Foi demonstrado que a meditação regular reduz hormônios do estresse, como o cortisol. Um estudo de 2018 publicado no Journal of Cognitive Enhancement descobriu que a meditação melhora a atenção e a memória de trabalho."
Justificativa : Alta relevância. O contexto aborda diretamente múltiplos benefícios para a saúde com detalhes específicos.

Exemplo (Ruim) :

Pergunta: "Quais são os benefícios da meditação para a saúde?"
Contexto: "As práticas de meditação variam muito entre as diferentes tradições. A meditação mindfulness, que teve origem nas práticas budistas, concentra-se na consciência do momento presente, enquanto a meditação transcendental utiliza mantras para alcançar estados de consciência mais profundos."
Justificativa : Baixa relevância. Apesar de apresentar informações factuais corretas sobre meditação, este contexto discute tipos e origens, em vez de benefícios para a saúde. O mecanismo de busca encontrou conteúdo relacionado ao tema, mas não a informação específica que ele buscava.
Contexto irrelevante condena geração

Se o seu mecanismo de recuperação extrair contexto irrelevante, o seu gerador estará fadado ao fracasso desde o início. Este é um erro comum, que reflete a "cegueira da ausência", onde as equipes se preocupam excessivamente com a qualidade da geração, enquanto negligenciam o funcionamento correto da recuperação (C|Q).

2. Fidelidade/Firmeza (A|C)

Definição : Em que medida a resposta se restringe apenas a afirmações que podem ser verificadas a partir do contexto recuperado? Isso avalia a capacidade do gerador de evitar alucinações.

Exemplo (Bom) :

Contexto: "A Grande Barreira de Corais é o maior sistema de recifes de coral do mundo."

Resposta: "A Grande Barreira de Corais é o maior sistema de recifes de coral do mundo."
Raciocínio : Fidelidade perfeita. A resposta apenas afirma o que está no contexto.

Exemplo (Ruim) :

Contexto: "A Grande Barreira de Corais é o maior sistema de recifes de coral do mundo. Ela se estende por mais de 2.300 quilômetros ao longo da costa de Queensland, na Austrália."

Resposta: "A Grande Barreira de Corais, o maior sistema de recifes de coral do mundo, estende-se por mais de 2.300 quilômetros ao longo da costa leste da Austrália e abriga cerca de 10% das espécies de peixes do mundo."
Raciocínio : Fidelidade mista. A primeira parte é corroborada, mas a afirmação sobre "10% das espécies de peixes do mundo" não se encontra no contexto fornecido. Essa sutil alucinação parece plausível e pode até ser factual, mas não está fundamentada no contexto apresentado.

Por que isso importa : Alucinações minam a confiança. É por isso que implementamos padrões de validação, citações interativas e raciocínio em cadeia em nossos aplicativos RAG — para detectar erros antes que cheguem aos usuários e construir confiança por meio da transparência.

3. Relevância da Resposta (A|Q)

Definição : Quão diretamente a resposta aborda a necessidade específica de informação expressa na pergunta? Isso avalia o desempenho do sistema de ponta a ponta.

Exemplo (Bom) :

Pergunta: "Como funcionam os juros compostos em investimentos?"

Resposta: "Os juros compostos funcionam adicionando os juros ganhos ao seu investimento inicial, de forma que os juros futuros sejam calculados sobre o novo valor, que será maior."
Raciocínio : Alta relevância. A resposta explica diretamente o conceito questionado.

Exemplo (Ruim) :

Pergunta: "Como funcionam os juros compostos em investimentos?"

Resposta: "Os juros em investimentos podem ser simples ou compostos. Os juros compostos são mais poderosos que os juros simples e são um conceito importante em finanças. É por isso que começar a investir cedo é tão benéfico para a construção de patrimônio a longo prazo."
Raciocínio : Baixa relevância. Apesar de tratar de juros compostos, a resposta não explica o mecanismo de funcionamento. Ela apenas menciona a importância do conceito, sem abordar a questão específica de como ele funciona.

Por que isso importa : Esta é a principal métrica de experiência do usuário. É também por isso que nos concentramos em criar mecanismos de feedback que perguntem especificamente "Respondemos à sua pergunta?" em vez de perguntas vagas como "Como nos saímos?". O feedback específico relacionado a essa métrica aumenta drasticamente as taxas de resposta.

Nível 3: Relações RAG avançadas¶
4. Cobertura de suporte contextual (C|A)

Definição : O contexto obtido contém todas as informações necessárias para sustentar plenamente cada afirmação na resposta? Isso mede se o contexto é suficiente e conciso.
Essa métrica corresponde ao que aprendemos sobre mecanismos de recuperação especializados e a arquitetura de roteamento de consultas. Diferentes tipos de conteúdo podem exigir abordagens de recuperação distintas para garantir uma cobertura completa. Por exemplo, ao responder perguntas sobre plantas de projetos de construção, pode ser necessário que tanto a recuperação de imagens quanto a recuperação de documentos trabalhem em conjunto.

5. Capacidade de resposta a perguntas (Q|C)

Definição : Dado o contexto fornecido, é realmente possível formular uma resposta satisfatória para a pergunta? Isso avalia se a pergunta é razoável considerando as informações disponíveis.
Isso corresponde ao padrão de rejeição estratégica que discutimos. Quando uma pergunta não pode ser respondida com o contexto disponível, a resposta mais honesta é reconhecer essa limitação. Isso constrói confiança por meio da transparência, em vez de gerar uma resposta fantasiosa.

6. Autocontenção (P|R)

Definição : A pergunta original pode ser inferida apenas a partir da resposta? Isso avalia se a resposta fornece contexto suficiente para ser compreendida por si só.
Isso está em consonância com nossa discussão sobre monólogos e abordagens de raciocínio lógico que tornam o pensamento visível. Respostas que reformulam e abordam a questão central diretamente criam melhores experiências para o usuário, especialmente em contextos de comunicação assíncrona.

Implementando a avaliação por níveis na prática¶
Com base em pesquisas acadêmicas recentes e experiência prática, veja como abordar a avaliação RAG com essas métricas:

Comece com o Nível 1 : Implemente métricas de recuperação rápida para o desenvolvimento diário.

Use precisão, recall, MAP@K e MRR@K para ajustar seu retriever.
Essas opções não exigem avaliação de mestrado em Direito (LLM) e oferecem ciclos de feedback rápidos.
Isso está em consonância com nossa abordagem de iniciar o ciclo de melhoria com dados sintéticos e métricas de avaliação específicas antes de passar para abordagens mais complexas.

Foco no Nível 2 : Implemente os três relacionamentos principais do modelo RAG.

Essas métricas principais (C|Q, A|C, A|Q) avaliam diretamente o desempenho do seu sistema RAG.
A maioria dos benchmarks prioriza essas três métricas.
Utilize a avaliação baseada em LLM para uma análise mais detalhada dessas relações.
Isso está alinhado com nosso foco em desenvolver mecanismos de feedback e melhorias na qualidade de vida que aumentem a confiança e a transparência.

Expanda para o Nível 3 : Adicione métricas avançadas quando precisar de insights mais detalhados.

Essas métricas (C|A, Q|C, Q|A) conectam o desempenho técnico aos resultados de negócios.
Utilize-os para avaliações mensais, lançamentos importantes e decisões estratégicas.
Diferentes domínios podem exigir ênfase em diferentes métricas de Nível 3 (por exemplo, o RAG médico precisa de um C|A mais robusto).
Isso está em consonância com nossa discussão sobre modelagem de tópicos e identificação de capacidades, reconhecendo que diferentes tipos de consulta podem exigir diferentes ênfases de avaliação.

Mestrado em Direito como Juiz¶
A maioria das avaliações modernas do RAG (Representação, Atitude e Qualidade) conta com mestres em Direito (LLM) como avaliadores. Essa abordagem, embora dispendiosa em termos de recursos, proporciona a avaliação mais precisa de nossas seis relações principais.

As nuances dos juízes do LLM

Embora demande muitos recursos, o uso de mestres em direito (LLMs) como avaliadores é atualmente o método mais eficaz para captar as nuances sutis das seis relações principais do modelo RAG. As métricas tradicionais geralmente se mostram insuficientes nessa avaliação complexa.

Diversos benchmarks, incluindo RAGAs, ARES e TruEra RAG Triad, agora utilizam a avaliação LLM por padrão. Embora métricas tradicionais como BLEU, ROUGE e BERTScore ainda tenham sua utilidade, somente a avaliação baseada em LLM consegue capturar efetivamente as nuances das relações em nossa estrutura.

Isso está em consonância com nossa discussão sobre o uso de Modelos de Aprendizagem Baseados em Aprendizagem (LLMs) para analisar feedback e identificar padrões em consultas de usuários – aproveitando a IA para entender a IA.

Avaliação específica do domínio¶
Uma descoberta interessante do benchmark DomainRAG é que diferentes domínios podem exigir ênfases diferentes dentro da nossa estrutura:

Os sistemas RAG médicos precisam de pontuações de fidelidade mais altas (A|C)
Atendimento ao cliente RAG exige maior relevância nas respostas (A|Q)
A documentação técnica RAG exige maior capacidade de resposta a perguntas (Q|C).
Isso está de acordo com o que aprendemos sobre modelagem de tópicos e segmentação: diferentes tipos de consulta exigem diferentes recursos, e nossa avaliação deve refletir essas prioridades. É por isso que segmentamos as perguntas não apenas por tópico, mas também pelo recurso necessário para respondê-las.

Por que essa estrutura é importante¶
Quando o seu sistema RAG falha, a falha ocorre em uma dessas dimensões. Sempre.

A resposta parece errada? Verifique a fidelidade (A|C).
A resposta parece irrelevante? Verifique a relevância da resposta (A|Q).
A resposta está faltando alguma informação importante? Verifique a relevância do contexto (C|Q) ou o suporte do contexto (C|A).
A beleza dessa estrutura reside em sua completude. Não existem outras relações entre Q, C e A. Abrangemos todos os ângulos de avaliação possíveis.

Essa abordagem sistemática para diagnosticar problemas está alinhada com a nossa mentalidade de produto para o modelo RAG – identificar pontos de falha específicos em vez de fazer afirmações vagas sobre "melhorar a IA".

E daí?¶
Na próxima vez que você estiver depurando um sistema RAG, não perca tempo com complexidade excessiva. Concentre-se nessas seis relações organizadas em níveis práticos. Corrija as que estiverem com defeito. Ignore o resto.

Essa estrutura está alinhada ao nosso ciclo de melhoria contínua: comece com o básico, colete feedback sobre aspectos específicos, analise esse feedback para identificar padrões e faça melhorias direcionadas com base no que você aprendeu.

E se alguém tentar lhe vender um modelo de avaliação RAG com 20 métricas diferentes? Sorria e pergunte qual das 6 relações principais eles estão realmente medindo.

Porque tanto na avaliação RAG quanto na implementação do RAG, a abordagem sistemática sempre se mostra vencedora.
