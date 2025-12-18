// Demo data module. Replace with real fetches as needed.
export const wellsGeoJSON = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      properties: {
        api: "33-105-12345",
        operator: "Example Oil LLC",
        formation: "Bakken",
        first_prod: "2018-06-12",
        cum_oil: 820000,
        timeseries: [
          { t: 0, q: 1200 },
          { t: 30, q: 980 },
          { t: 60, q: 860 },
          { t: 90, q: 780 },
          { t: 120, q: 720 },
          { t: 150, q: 670 },
          { t: 180, q: 630 },
          { t: 210, q: 600 },
          { t: 240, q: 575 },
          { t: 270, q: 555 },
          { t: 300, q: 540 },
          { t: 330, q: 525 },
          { t: 360, q: 510 }
        ]
      },
      geometry: { type: "Point", coordinates: [-103.6, 47.8] }
    },
    {
      type: "Feature",
      properties: {
        api: "33-061-67890",
        operator: "Prairie Resources",
        formation: "Three Forks",
        first_prod: "2020-03-05",
        cum_oil: 410000,
        timeseries: [
          { t: 0, q: 900 },
          { t: 30, q: 770 },
          { t: 60, q: 700 },
          { t: 90, q: 650 },
          { t: 120, q: 610 },
          { t: 150, q: 575 },
          { t: 180, q: 545 },
          { t: 210, q: 520 },
          { t: 240, q: 500 }
        ]
      },
      geometry: { type: "Point", coordinates: [-102.9, 48.2] }
    }
  ]
};
