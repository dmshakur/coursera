using DelimitedFiles
using Plots

evd_data = DelimitedFiles.readdlm("wikipediaEVDdatesconverted.csv", ',')
epidays = evd_data[:, 1]
allcases = evd_data[:, 2]

gr()
plot(epidays, allcases)
plot(epidays, allcases, linetype = :scatter, marker = :diamond)

plot(epidays, allcases,
    title` = "west african evd epidemic, total cases",
    xlabel = "days since 22 march 2014",
    ylabel = "total cases to date (three countries)",
    marker = (:diamond, 8),
    line = (:path, "gray"),
    legend = false,
    grid = false)

savefig('wafrican_evd_noformatspecified')
savefig('wafrican_evd_noformatspecified.pdf')
savefig('wafrican_evd_noformatspecified.png')
