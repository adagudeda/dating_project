# Анализ дейтинг-приложения

## Задача
Проанализировать изменение стоимости премиум-подписки на конкретные группы пользователей и дать свою оценку относительно успешности эксперимента.

## Резуальтаты исследования
Изначально был проведен A/A тест для проверки качества сплитования, убедившись в репрезентативности двух групп, был проведен анализ ключевых метрик по конкретным группам
пользователей. При сравнении тестовой группы с контрольной группой было установлено, что результатом эксперимента стало статистически значимое увеличение ARPPU (34,65%), которое, однако, сопровождалось значимым уменьшением конверсии в покупку. Дополнительный анализ по ARPU показал, что статистически значимого прироста среднего чека нет. Изменение стоимости подписки не рекомендуется к введению для всех групп пользователей, так как не даст значимого прироста прибыли

## Использованные библиотеки
- *pandas*
- *numpy*
- *scipy*
- *pingouin*
- *matplotlib*
- *seaborn*
- *requests*
