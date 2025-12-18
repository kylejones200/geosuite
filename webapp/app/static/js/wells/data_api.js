// API data module - connects to Flask backend
// This replaces static mock data with real API calls

export async function fetchWellsData() {
    try {
        console.log('Fetching wells data from API...');
        const response = await fetch('/wells/api/wells');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Wells data loaded:', data);
        return data;
    } catch (error) {
        console.error('Error fetching wells data:', error);
        
        // Fallback to mock data if API fails
        return {
            type: "FeatureCollection",
            features: [
                {
                    id: "33053043310000",
                    type: "Feature", 
                    properties: {
                        api: "33053043310000",
                        well_name: "BAKKEN 1H",
                        operator: "Continental Resources",
                        formation: "BAKKEN",
                        first_prod: "2020-01",
                        cum_oil: 250000
                    },
                    geometry: { type: "Point", coordinates: [-103.2, 47.8] }
                }
            ]
        };
    }
}

export async function fetchOperatorsData(since = null) {
    try {
        console.log('Fetching operators data from API...');
        const url = since ? `/wells/api/operators?since=${since}` : '/wells/api/operators';
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Operators data loaded:', data);
        return data;
    } catch (error) {
        console.error('Error fetching operators data:', error);
        
        // Fallback to mock data if API fails
        return {
            operators: [
                {
                    operator: "Continental Resources",
                    wells: 150,
                    avg_qi: 800,
                    total_cum_oil: 15000000
                },
                {
                    operator: "Whiting Petroleum",
                    wells: 120, 
                    avg_qi: 750,
                    total_cum_oil: 12000000
                }
            ]
        };
    }
}

export async function fetchWellDetail(uwi) {
    try {
        console.log(`Fetching well detail for UWI: ${uwi}`);
        const response = await fetch(`/wells/api/wells/${uwi}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Well detail loaded:', data);
        return data;
    } catch (error) {
        console.error('Error fetching well detail:', error);
        
        // Fallback mock data
        return {
            api: uwi,
            well_name: "BAKKEN 1H",
            operator: "Continental Resources",
            formation: "BAKKEN",
            first_prod: "2020-01", 
            cum_oil: 250000,
            timeseries: [
                { t: 30, q: 800 },
                { t: 60, q: 650 },
                { t: 90, q: 520 }
            ],
            water_rates: [50, 45, 40],
            gas_rates: [200, 180, 160]
        };
    }
}
