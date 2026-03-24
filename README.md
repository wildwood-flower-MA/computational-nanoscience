# Obliczeniowa mechanika kwantowa

To repozytorium zawiera implementacje kilku różnych metod numerycznych służących do rozwiązywania problemów kwantowo-mechanicznych opartych na równaniu Schrödingera:

$$\left[ -\frac{\hbar^2}{2m} \nabla^2 + V(x) \right] \psi(x, t) = \text{i}\hbar \frac{\partial}{\partial t} \psi(x, t).$$

Zazwyczaj głównym celem jest znalezienie wartości własnych (energii) i funkcji własnych hamiltonianu dla zadanych układów nanostrukturalnych. Rozważane są też zagadnienia transportu kwantowego. Z grubsza zagadnienia i metody to:

- *Stacjonarne równanie Schrödingera:* dyskretyzacja na siatce (FDM), rozwiązywanie wielkoskalowych problemów własnych.
- *Równanie Schrödingera zależne od czasu:* ewolucja czasowa układów kwantowych, symulacja dynamiki pakietów falowych (np. propagatory, metoda Cranka-Nicolson).
- *Potencjały modelowe:* analiza stanów w m.in. podwójnej studni potencjału (QD), barierach tunelowych i oscylatorach.
- *Transport kwantowy:* wyznaczanie współczynników transmisji i odbicia, tunelowanie rezonansowe.
