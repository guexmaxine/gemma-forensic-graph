import sys
sys.stdout.reconfigure(encoding='utf-8')
from entity_extractor_v2 import ForensicEntityTaxonomyExtractor

sample_text = """
[2022-08-09 21:31:28] Cliente: Consegue manda foto do imei pra mim consulta no site da Apple?
[2022-08-09 21:31:37] Vendedor: Aí fico com um iPhone 11 Pro Gold.
[2022-08-09 21:34:18] Vendedor: Aí vou ficar lhe devendo imei e um número de série do aparelho através desse número vc consegue bloquear o aparelho portanto se torna sigiloso ao dono ou futuro dono.
[2022-08-09 21:35:40] Cliente: Fiz o pix de 2300 pra conta do João Batista que o Tito passou ontem à tarde.
[2022-08-09 21:36:01] Vendedor: Me chama no privado que tiro todas as suas dúvidas com o Gordo da van.
[2022-11-23 17:23:41] Fabian: Receba R$ 50,00 agora seu PIX NATALINO da Promoção Nubank em Vitória!
"""

ents = ForensicEntityTaxonomyExtractor.extract_entities_from_text(sample_text)
print(f"Total entities detected: {len(ents)}")
for e in ents:
    print(f"  [{e['categoria']}] '{e['text']}' (conf: {e['confidence']})")
