# Setting safety stock policy across product segments

A short case note for an operations manager deciding how much buffer stock to hold, and
where.

## The situation

You run replenishment for a distribution business carrying around a hundred active SKUs.
Seven of them produce 80% of your revenue. The current rule is the one most planners
inherit: hold a buffer sized on how variable demand has been, and treat the delivery lead
time as if it were a fixed number of days.

The question in front of you is whether that rule is costing you, and if so, where.

## What I found

**Your buffer is being sized against the wrong source of uncertainty on the products that
matter most.**

Safety stock has to absorb two things: demand arriving faster than expected, and
replenishment arriving later than expected. The standard rule only handles the first. For
your top seven products, the second is where roughly seven eighths of the total uncertainty
actually sits.

| Product group | Share of revenue | Uncertainty from lead time | Uncertainty from demand |
|---|---|---|---|
| Top 7 (high volume, steady) | 82% | 88% | 12% |
| Slow moving tail (29 products) | 2% | 10% | 90% |

The reason is arithmetic rather than subtle. A product selling 70 units a day with steady
demand loses very little to demand noise over a four day lead time. But every extra day of
delay costs it another 70 units of exposure. A product selling under one unit a day is the
other way round: a day of delay barely matters, while whether this week brings four orders
or none matters a great deal.

So the same policy rule cannot be right for both ends of your catalogue, and the rule you
are running is wrong at the end that carries your revenue.

## What it costs

I simulated three policies over a nine month period the parameters were not fitted on,
using your actual order history.

| Policy | Units of demand lost | Service achieved | Annual holding cost |
|---|---|---|---|
| Current approach (demand variability only) | 595 | 82% | 45,239 |
| Flat 50% buffer on everything | 149 | 90% | 47,243 |
| Segmented, both sources of uncertainty | 8 | 98% | 50,820 |

The current approach is set to a 98% service target on the A items and delivers 82%. That
is not a small miss. It is the policy failing to do the thing it was configured to do,
because the formula behind it does not account for the delay risk that dominates those
products.

Closing that gap costs about 6,300 a year in extra carrying cost across the book and
recovers 587 units of demand that currently walk out the door.

Worth noting for credibility: on the slow moving tail, the current approach and the
recommended one are almost identical. The recommendation is not "spend more everywhere". It
is "spend more on the seven products where delay risk is the binding constraint, and leave
the tail alone".

## What I recommend

**Differentiate the targets, and be explicit about why.**

| Group | Products | Target | Reasoning |
|---|---|---|---|
| A, high volume steady | 7 | 98% | Buffer costs 1.38 per 1,000 of revenue protected. Cheapest service you will ever buy. |
| B, mid volume steady | 2 | 95% | Buffer costs 1.12 per 1,000. Similar economics, lower absolute exposure. |
| Erratic tail | 31 | 90% | Buffer costs 2.55 to 3.17 per 1,000. Two to three times more expensive per unit of revenue defended. |

The tail gets a lower target not because those products matter less in principle, but
because protecting them is measurably more expensive per pound of revenue protected. Erratic
demand needs disproportionately more buffer for each additional point of service.

**Do not chase the last two points of service.** For your largest product, moving from 80%
to 98% costs about 1,240 a year. Moving from 98% to 99.9% costs another 1,060 on its own,
and the marginal cost per point of service rises from 40 to over 1,300 across that stretch.
Somewhere past 98% you are buying insurance at a price nobody has justified, because I do
not have a number for what a stockout actually costs you. If you can give me that number, I
can turn this tradeoff into an optimum. Until then, 98% is where the curve turns.

**Fix the delivery time variability instead of buying more buffer for it.** This is the
recommendation with the largest prize attached. The top seven products need roughly 2.9
times the safety stock they would need under a reliable lead time, purely because delivery
timing varies. If a fixed lead time could be guaranteed, the buffer across the whole book
would fall from about 1,110 units to about 485, a 56% reduction, without giving up any
service. That is a conversation with your carrier or your supplier, not with your planning
system, and it is worth more than any parameter change I can make.

## What I would want before acting on this

Three things, in order of how much they would change the answer.

**A stockout cost.** Without it, every service target here is a judgment call dressed up in
a curve. With it, the targets become a calculation.

**Real inbound lead time data.** The delivery times I could observe are outbound, customer
facing. Inbound replenishment lead time, supplier to warehouse, is what the safety stock
formula actually needs, and it is not in the data I was given. I have used the outbound
distribution as a stand-in. The structure of the recommendation is robust to that
substitution, and I checked: the conclusion that delay risk dominates on the A items holds
across every plausible value of the delay variability. But the specific unit counts would
move.

**A tiered ordering cost.** The order quantity calculation assumes it costs the same to
raise a purchase order for a product selling 70 a day and one selling 0.8 a day. That
assumption gives the slow movers a recommended order size worth seven months of cover,
which is obviously not how you would run them. The fast movers come out at two to three
weeks, which is sensible. Fixing the tail needs either a per-product ordering cost or a
maximum cover constraint.

## The one line version

Your service problem on the products that matter is not a demand forecasting problem, it is
a delivery reliability problem, and your current safety stock formula cannot see it. Sizing
the buffer for both sources of uncertainty closes an 82% to 98% service gap for about 6,300
a year. Making the deliveries reliable in the first place would let you take half the buffer
back out.
