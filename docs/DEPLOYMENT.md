## What approach would you use to version and track different models in production?
Using model registries like MLFlow or Azure ML that allow the control of the full lifespan of the model allowing data storage with metadata, versioning, use of aliases, tracking the experiment and run that produced the model, different stages in development, staging and production and rollbacks to previous versions, both for code and models.

## What key metrics would you monitor for this API service and the prediction model?
Regarding the API service I will focus on the availability of the service by checking whether its availability, latency percetage, error rates, resource usage, traffic and saturation. On the other hand for the prediction model I would check periodically ground-truth performance on sepparated environments using real data, the quality of the newer data, online performance and operational metrics as latency.

## How would you roll back to a previous version if needed?
Keeping three stages with staging, production and archive, I would demote the current model that was failing in production, promote the last stable model from archive and send it into production. Then I would check which of the two models perform better comparing with the same inputs in a safe environment out of production and try to find and the cause of the error on the first one and compare which model performs better, rather keeping the older version or repromoting the newer one.
